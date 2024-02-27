//!
//! The layer map tracks what layers exist in a timeline.
//!
//! When the timeline is first accessed, the server lists of all layer files
//! in the timelines/<timeline_id> directory, and populates this map with
//! ImageLayer and DeltaLayer structs corresponding to each file. When the first
//! new WAL record is received, we create an InMemoryLayer to hold the incoming
//! records. Now and then, in the checkpoint() function, the in-memory layer is
//! are frozen, and it is split up into new image and delta layers and the
//! corresponding files are written to disk.
//!
//! Design overview:
//!
//! The `search` method of the layer map is on the read critical path, so we've
//! built an efficient data structure for fast reads, stored in `LayerMap::historic`.
//! Other read methods are less critical but still impact performance of background tasks.
//!
//! This data structure relies on a persistent/immutable binary search tree. See the
//! following lecture for an introduction <https://www.youtube.com/watch?v=WqCWghETNDc&t=581s>
//! Summary: A persistent/immutable BST (and persistent data structures in general) allows
//! you to modify the tree in such a way that each modification creates a new "version"
//! of the tree. When you modify it, you get a new version, but all previous versions are
//! still accessible too. So if someone is still holding a reference to an older version,
//! they continue to see the tree as it was then. The persistent BST stores all the
//! different versions in an efficient way.
//!
//! Our persistent BST maintains a map of which layer file "covers" each key. It has only
//! one dimension, the key. See `layer_coverage.rs`. We use the persistent/immutable property
//! to handle the LSN dimension.
//!
//! To build the layer map, we insert each layer to the persistent BST in LSN.start order,
//! starting from the oldest one. After each insertion, we grab a reference to that "version"
//! of the tree, and store it in another tree, a BtreeMap keyed by the LSN. See
//! `historic_layer_coverage.rs`.
//!
//! To search for a particular key-LSN pair, you first look up the right "version" in the
//! BTreeMap. Then you search that version of the BST with the key.
//!
//! The persistent BST keeps all the versions, but there is no way to change the old versions
//! afterwards. We can add layers as long as they have larger LSNs than any previous layer in
//! the map, but if we need to remove a layer, or insert anything with an older LSN, we need
//! to throw away most of the persistent BST and build a new one, starting from the oldest
//! LSN. See [`LayerMap::flush_updates()`].
//!

mod historic_layer_coverage;
mod layer_coverage;

use crate::context::RequestContext;
use crate::keyspace::KeyPartitioning;
use crate::repository::Key;
use crate::tenant::storage_layer::InMemoryLayer;
use anyhow::Result;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use utils::lsn::Lsn;

use historic_layer_coverage::BufferedHistoricLayerCoverage;
pub use historic_layer_coverage::LayerKey;

use super::storage_layer::PersistentLayerDesc;

///
/// LayerMap tracks what layers exist on a timeline.
///
#[derive(Default)]
pub struct LayerMap {
    //
    // 'open_layer' holds the current InMemoryLayer that is accepting new
    // records. If it is None, 'next_open_layer_at' will be set instead, indicating
    // where the start LSN of the next InMemoryLayer that is to be created.
    //
    pub open_layer: Option<Arc<InMemoryLayer>>,
    pub next_open_layer_at: Option<Lsn>,

    ///
    /// Frozen layers, if any. Frozen layers are in-memory layers that
    /// are no longer added to, but haven't been written out to disk
    /// yet. They contain WAL older than the current 'open_layer' or
    /// 'next_open_layer_at', but newer than any historic layer.
    /// The frozen layers are in order from oldest to newest, so that
    /// the newest one is in the 'back' of the VecDeque, and the oldest
    /// in the 'front'.
    ///
    pub frozen_layers: VecDeque<Arc<InMemoryLayer>>,

    /// Index of the historic layers optimized for search
    historic: BufferedHistoricLayerCoverage<Arc<PersistentLayerDesc>>,

    /// L0 layers have key range Key::MIN..Key::MAX, and locating them using R-Tree search is very inefficient.
    /// So L0 layers are held in l0_delta_layers vector, in addition to the R-tree.
    ///
    /// XI: It seems L0 layer only store the descriptors of the layers, not the actual data.
    ///     And based on LSM-Tree knowledge, the L0 layer's descriptors may have key-range overlap with each other.
    l0_delta_layers: Vec<Arc<PersistentLayerDesc>>,
}

/// The primary update API for the layer map.
///
/// Batching historic layer insertions and removals is good for
/// performance and this struct helps us do that correctly.
#[must_use]
pub struct BatchedUpdates<'a> {
    // While we hold this exclusive reference to the layer map the type checker
    // will prevent us from accidentally reading any unflushed updates.
    layer_map: &'a mut LayerMap,
}

/// Provide ability to batch more updates while hiding the read
/// API so we don't accidentally read without flushing.
impl BatchedUpdates<'_> {
    ///
    /// Insert an on-disk layer.
    ///
    // TODO remove the `layer` argument when `mapping` is refactored out of `LayerMap`
    pub fn insert_historic(&mut self, layer_desc: PersistentLayerDesc) {
        self.layer_map.insert_historic_noflush(layer_desc)
    }

    ///
    /// Remove an on-disk layer from the map.
    ///
    /// This should be called when the corresponding file on disk has been deleted.
    ///
    pub fn remove_historic(&mut self, layer_desc: &PersistentLayerDesc) {
        self.layer_map.remove_historic_noflush(layer_desc)
    }

    // We will flush on drop anyway, but this method makes it
    // more explicit that there is some work being done.
    /// Apply all updates
    pub fn flush(self) {
        // Flush happens on drop
    }
}

// Ideally the flush() method should be called explicitly for more
// controlled execution. But if we forget we'd rather flush on drop
// than panic later or read without flushing.
//
// TODO maybe warn if flush hasn't explicitly been called
impl Drop for BatchedUpdates<'_> {
    fn drop(&mut self) {
        self.layer_map.flush_updates();
    }
}

/// Return value of LayerMap::search
pub struct SearchResult {
    pub layer: Arc<PersistentLayerDesc>,
    pub lsn_floor: Lsn,
}

impl LayerMap {
    ///
    /// Find the latest layer (by lsn.end) that covers the given
    /// 'key', with lsn.start < 'end_lsn'.
    ///
    /// The caller of this function is the page reconstruction
    /// algorithm looking for the next relevant delta layer, or
    /// the terminal image layer. The caller will pass the lsn_floor
    /// value as end_lsn in the next call to search.
    ///
    /// If there's an image layer exactly below the given end_lsn,
    /// search should return that layer regardless if there are
    /// overlapping deltas.
    ///
    /// If the latest layer is a delta and there is an overlapping
    /// image with it below, the lsn_floor returned should be right
    /// above that image so we don't skip it in the search. Otherwise
    /// the lsn_floor returned should be the bottom of the delta layer
    /// because we should make as much progress down the lsn axis
    /// as possible. It's fine if this way we skip some overlapping
    /// deltas, because the delta we returned would contain the same
    /// wal content.
    ///
    /// TODO: This API is convoluted and inefficient. If the caller
    /// makes N search calls, we'll end up finding the same latest
    /// image layer N times. We should either cache the latest image
    /// layer result, or simplify the api to `get_latest_image` and
    /// `get_latest_delta`, and only call `get_latest_image` once.
    ///
    /// NOTE: This only searches the 'historic' layers, *not* the
    /// 'open' and 'frozen' layers!
    ///
    pub fn search(&self, key: Key, end_lsn: Lsn) -> Option<SearchResult> {
        // XI: The latest version that start.lsn < end_lsn
        let version = self.historic.get().unwrap().get_version(end_lsn.0 - 1)?;
        let latest_delta = version.delta_coverage.query(key.to_i128());
        let latest_image = version.image_coverage.query(key.to_i128());

        match (latest_delta, latest_image) {
            (None, None) => None,
            // XI: if there is only an image layer, return this image layer descriptor and its lsn
            (None, Some(image)) => {
                let lsn_floor = image.get_lsn_range().start;
                Some(SearchResult {
                    layer: image,
                    lsn_floor,
                })
            }
            // XI: If this is a delta layer, return this delta layer descriptor and its start lsn
            (Some(delta), None) => {
                let lsn_floor = delta.get_lsn_range().start;
                Some(SearchResult {
                    layer: delta,
                    lsn_floor,
                })
            }
            // XI: If there is both a delta layer and an image layer, return the newer one
            //     If image layer is newer, just return it normally
            //     If delta layer is newer, return it and return the floor lsn greater than the image layer's start lsn
            //        In this way, the image layer can be searched again in the next search
            (Some(delta), Some(image)) => {
                //XI: Image layer's lsn range is [lsn, lsn+1)
                let img_lsn = image.get_lsn_range().start;
                let image_is_newer = image.get_lsn_range().end >= delta.get_lsn_range().end;
                let image_exact_match = img_lsn + 1 == end_lsn;
                if image_is_newer || image_exact_match {
                    Some(SearchResult {
                        layer: image,
                        lsn_floor: img_lsn,
                    })
                } else {
                    let lsn_floor =
                        std::cmp::max(delta.get_lsn_range().start, image.get_lsn_range().start + 1);
                    Some(SearchResult {
                        layer: delta,
                        lsn_floor,
                    })
                }
            }
        }
    }

    /// Start a batch of updates, applied on drop
    pub fn batch_update(&mut self) -> BatchedUpdates<'_> {
        BatchedUpdates { layer_map: self }
    }

    ///
    /// Insert an on-disk layer
    ///
    /// Helper function for BatchedUpdates::insert_historic
    ///
    /// TODO(chi): remove L generic so that we do not need to pass layer object.
    /// XI: Insert this layer into the historic layer coverage.
    ///     If this is an L0 layer, also insert it into l0_delta_layers.
    ///     Question: How to understand the L0 layer? Is this logical layer include several physical layer descriptors?
    pub(self) fn insert_historic_noflush(&mut self, layer_desc: PersistentLayerDesc) {
        // TODO: See #3869, resulting #4088, attempted fix and repro #4094

        // XI: For the L0 layer, there are no exact key range, which means the
        //     the key.start==const.KEY_MIN and key.end==const.KEY_MAX
        if Self::is_l0(&layer_desc) {
            self.l0_delta_layers.push(layer_desc.clone().into());
        }

        // XI: this insert operation will be temporarily cached in the historic layer coverage.
        self.historic.insert(
            historic_layer_coverage::LayerKey::from(&layer_desc),
            layer_desc.into(),
        );
    }

    ///
    /// Remove an on-disk layer from the map.
    ///
    /// Helper function for BatchedUpdates::remove_historic
    ///
    /// XI: Remove this layer from historic layer coverage.
    ///     If this is a L0 layer, also remove it from l0_delta_layers.
    pub fn remove_historic_noflush(&mut self, layer_desc: &PersistentLayerDesc) {
        // XI: This remove operation will temporarily be cached in the historic layer coverage.
        self.historic
            .remove(historic_layer_coverage::LayerKey::from(layer_desc));
        let layer_key = layer_desc.key();
        // XI: If this is a L0 layer, remove it from l0_delta_layers
        if Self::is_l0(layer_desc) {
            let len_before = self.l0_delta_layers.len();
            let mut l0_delta_layers = std::mem::take(&mut self.l0_delta_layers);
            l0_delta_layers.retain(|other| other.key() != layer_key);
            self.l0_delta_layers = l0_delta_layers;
            // this assertion is related to use of Arc::ptr_eq in Self::compare_arced_layers,
            // there's a chance that the comparison fails at runtime due to it comparing (pointer,
            // vtable) pairs.
            assert_eq!(
                self.l0_delta_layers.len(),
                len_before - 1,
                "failed to locate removed historic layer from l0_delta_layers"
            );
        }
    }

    /// Helper function for BatchedUpdates::drop.
    /// XI: Before calling flush_updates, the updating operations are cached in historic's buffer.
    ///     By calling this function, the buffer will be flushed to Persistent BST
    pub(self) fn flush_updates(&mut self) {
        self.historic.rebuild();
    }

    /// Is there a newer image layer for given key- and LSN-range? Or a set
    /// of image layers within the specified lsn range that cover the entire
    /// specified key range?
    ///
    /// This is used for garbage collection, to determine if an old layer can
    /// be deleted.
    ///
    /// XI: For a specific LSN, there is a version of Persistent BST.
    /// For this specific version, there is one image_coverage and one delta_coverage.
    /// one image_coverage doesn't mean these layers share same LSN. In fact, there are layers with
    /// different key-range and corresponding different LSNs.
    pub fn image_layer_exists(&self, key: &Range<Key>, lsn: &Range<Lsn>) -> Result<bool> {
        if key.is_empty() {
            // Vacuously true. There's a newer image for all 0 of the kerys in the range.
            return Ok(true);
        }

        //XI: Get the latest version that lsn_range.end_lsn < lsn.end
        let version = match self.historic.get().unwrap().get_version(lsn.end.0 - 1) {
            Some(v) => v,
            None => return Ok(false),
        };

        let start = key.start.to_i128();
        let end = key.end.to_i128();

        //XI: This Closure will check whether one layer's start LSN is greater than or equal to the given LSN
        //    Question, why don't need to check get_lsn_range().start < lsn.end?
        //    Perhaps, this function's working scenario is: it found a old image layer, and wanna know whether this
        //    layer can be deleted. So, it will check whether these is a layer newer than this old image layer(only need
        //    to check the parameter lsn_range.start), and also cover this old image layer's key range.
        let layer_covers = |layer: Option<Arc<PersistentLayerDesc>>| match layer {
            Some(layer) => layer.get_lsn_range().start >= lsn.start,
            None => false,
        };


        // XI: Parameter key_range = [100, 300]
        //     image_coverage: [90-110, 110-130, 130-190, 190-220]

        // Check the start is covered
        //XI: version.image_coverage.query(start) will find the last layer that key.start <= start
        //    If one layer's key.start > start, it means this layer's key range can't cover the parameter key range
        //    Then check whether this layer's start LSN is greater than parameter lsn_range.start
        //    Either of the two conditions is not satisfied, return false
        //    This check actually check [90-110]'s lsn range
        if !layer_covers(version.image_coverage.query(start)) {
            return Ok(false);
        }

        // XI: In fact, this parameter key-range may consist of several sub-ranges, and each sub-range
        //     should be checked. Any sub-range doesn't satisfy the LSN condition, return false
        //     This for loop will check [110-130, 130-190, 190-220]

        // Check after all changes of coverage
        for (_, change_val) in version.image_coverage.range(start..end) {
            if !layer_covers(change_val) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    //XI: Iterate the current layer coverage
    pub fn iter_historic_layers(&self) -> impl '_ + Iterator<Item = Arc<PersistentLayerDesc>> {
        self.historic.iter()
    }

    ///
    /// Divide the whole given range of keys into sub-ranges based on the latest
    /// image layer that covers each range at the specified lsn (inclusive).
    /// This is used when creating  new image layers.
    ///
    // FIXME: clippy complains that the result type is very complex. She's probably
    // right...
    #[allow(clippy::type_complexity)]
    pub fn image_coverage(
        &self,
        key_range: &Range<Key>,
        lsn: Lsn,
    ) -> Result<Vec<(Range<Key>, Option<Arc<PersistentLayerDesc>>)>> {
        let version = match self.historic.get().unwrap().get_version(lsn.0) {
            Some(v) => v,
            None => return Ok(vec![]),
        };

        //XI: Let's assume the key_range is [100, 300]
        //    The image_coverage is [90-110, 110-130, 130-190, 190-220]
        let start = key_range.start.to_i128();
        let end = key_range.end.to_i128();

        // Initialize loop variables
        let mut coverage: Vec<(Range<Key>, Option<Arc<PersistentLayerDesc>>)> = vec![];
        let mut current_key = start;
        let mut current_val = version.image_coverage.query(start);
        //XI: Up to now, the current_key = 100, current_val = [90-110]

        //XI: For the first loop: change_key = 110, change_val = [110-130]
        //    kr = 100..110, coverage.push([90-110])
        //    current_key = 110, current_val = [110-130]
        //    For the second loop: change_key = 130, change_val = [130-190]
        //    kr = 110..130, coverage.push([110-130])
        //    current_key = 130, current_val = [130-190]
        //    So on and so forth ...

        // Loop through the change events and push intervals
        for (change_key, change_val) in version.image_coverage.range(start..end) {
            let kr = Key::from_i128(current_key)..Key::from_i128(change_key);
            coverage.push((kr, current_val.take()));
            current_key = change_key;
            current_val = change_val.clone();
        }

        // Add the final interval
        let kr = Key::from_i128(current_key)..Key::from_i128(end);
        coverage.push((kr, current_val.take()));

        Ok(coverage)
    }

    pub fn is_l0(layer: &PersistentLayerDesc) -> bool {
        layer.get_key_range() == (Key::MIN..Key::MAX)
    }

    /// This function determines which layers are counted in `count_deltas`:
    /// layers that should count towards deciding whether or not to reimage
    /// a certain partition range.
    ///
    /// There are two kinds of layers we currently consider reimage-worthy:
    ///
    /// Case 1: Non-L0 layers are currently reimage-worthy by default.
    /// TODO Some of these layers are very sparse and cover the entire key
    ///      range. Replacing 256MB of data (or less!) with terabytes of
    ///      images doesn't seem wise. We need a better heuristic, possibly
    ///      based on some of these factors:
    ///      a) whether this layer has any wal in this partition range
    ///      b) the size of the layer
    ///      c) the number of images needed to cover it
    ///      d) the estimated time until we'll have to reimage over it for GC
    ///
    /// Case 2: Since L0 layers by definition cover the entire key space, we consider
    /// them reimage-worthy only when the entire key space can be covered by very few
    /// images (currently 1).
    /// TODO The optimal number should probably be slightly higher than 1, but to
    ///      implement that we need to plumb a lot more context into this function
    ///      than just the current partition_range.
    pub fn is_reimage_worthy(layer: &PersistentLayerDesc, partition_range: &Range<Key>) -> bool {
        // Case 1
        if !Self::is_l0(layer) {
            return true;
        }

        // Case 2
        if partition_range == &(Key::MIN..Key::MAX) {
            return true;
        }

        false
    }

    /// Count the height of the tallest stack of reimage-worthy deltas
    /// in this 2d region.
    ///
    /// If `limit` is provided we don't try to count above that number.
    ///
    /// This number is used to compute the largest number of deltas that
    /// we'll need to visit for any page reconstruction in this region.
    /// We use this heuristic to decide whether to create an image layer.
    ///
    /// XI: Using the recursion to count the number of delta layers with provided key and lsn range
    ///  To understand this function, firstly, for one specific key, there may be several delta layers.
    ///  Each delta layer stands for a version between [lsn_start, lsn_end).
    ///  So, this function will firstly find the lasted version smaller(elder) than lsn.end.
    ///  Let assume, this version's lsn-range is [100, 120) and parameter lsn range is [30, 110),
    ///     then the result equal to "count_deltas( [30, 100) ) +1" -> Recursion function
    ///
    ///  But remember, the delta_coverage consists of several sub-ranges, and each sub-range is a delta layer.
    ///  So, for each sub-range, it will call count_deltas recursively. And the result is the maximum value of all sub-ranges.
    pub fn count_deltas(
        &self,
        key: &Range<Key>,
        lsn: &Range<Lsn>,
        limit: Option<usize>,
    ) -> Result<usize> {
        // We get the delta coverage of the region, and for each part of the coverage
        // we recurse right underneath the delta. The recursion depth is limited by
        // the largest result this function could return, which is in practice between
        // 3 and 10 (since we usually try to create an image when the number gets larger).

        if lsn.is_empty() || key.is_empty() || limit == Some(0) {
            return Ok(0);
        }

        let version = match self.historic.get().unwrap().get_version(lsn.end.0 - 1) {
            Some(v) => v,
            None => return Ok(0),
        };

        let start = key.start.to_i128();
        let end = key.end.to_i128();

        // Initialize loop variables
        let mut max_stacked_deltas = 0;
        let mut current_key = start;
        let mut current_val = version.delta_coverage.query(start);

        // Loop through the delta coverage and recurse on each part
        for (change_key, change_val) in version.delta_coverage.range(start..end) {
            // If there's a relevant delta in this part, add 1 and recurse down
            if let Some(val) = current_val {
                //XI: val.get_lsn_range().end > lsn.start means this layer would be accessed during reconstructing
                if val.get_lsn_range().end > lsn.start {
                    let kr = Key::from_i128(current_key)..Key::from_i128(change_key);
                    let lr = lsn.start..val.get_lsn_range().start;
                    if !kr.is_empty() {
                        //XI: Need to understand the meaning of is_reimage_worthy()
                        let base_count = Self::is_reimage_worthy(&val, key) as usize;
                        let new_limit = limit.map(|l| l - base_count);
                        let max_stacked_deltas_underneath =
                            self.count_deltas(&kr, &lr, new_limit)?;
                        max_stacked_deltas = std::cmp::max(
                            max_stacked_deltas,
                            base_count + max_stacked_deltas_underneath,
                        );
                    }
                }
            }

            current_key = change_key;
            current_val = change_val.clone();
        }

        // Consider the last part
        if let Some(val) = current_val {
            if val.get_lsn_range().end > lsn.start {
                let kr = Key::from_i128(current_key)..Key::from_i128(end);
                let lr = lsn.start..val.get_lsn_range().start;

                if !kr.is_empty() {
                    let base_count = Self::is_reimage_worthy(&val, key) as usize;
                    let new_limit = limit.map(|l| l - base_count);
                    let max_stacked_deltas_underneath = self.count_deltas(&kr, &lr, new_limit)?;
                    max_stacked_deltas = std::cmp::max(
                        max_stacked_deltas,
                        base_count + max_stacked_deltas_underneath,
                    );
                }
            }
        }

        Ok(max_stacked_deltas)
    }

    /// Count how many reimage-worthy layers we need to visit for given key-lsn pair.
    ///
    /// The `partition_range` argument is used as context for the reimage-worthiness decision.
    ///
    /// Used as a helper for correctness checks only. Performance not critical.
    pub fn get_difficulty(&self, lsn: Lsn, key: Key, partition_range: &Range<Key>) -> usize {
        match self.search(key, lsn) {
            Some(search_result) => {
                if search_result.layer.is_incremental() {
                    (Self::is_reimage_worthy(&search_result.layer, partition_range) as usize)
                        + self.get_difficulty(search_result.lsn_floor, key, partition_range)
                } else {
                    0
                }
            }
            None => 0,
        }
    }

    /// Used for correctness checking. Results are expected to be identical to
    /// self.get_difficulty_map. Assumes self.search is correct.
    pub fn get_difficulty_map_bruteforce(
        &self,
        lsn: Lsn,
        partitioning: &KeyPartitioning,
    ) -> Vec<usize> {
        // Looking at the difficulty as a function of key, it could only increase
        // when a delta layer starts or an image layer ends. Therefore it's sufficient
        // to check the difficulties at:
        // - the key.start for each non-empty part range
        // - the key.start for each delta
        // - the key.end for each image
        let keys_iter: Box<dyn Iterator<Item = Key>> = {
            let mut keys: Vec<Key> = self
                .iter_historic_layers()
                .map(|layer| {
                    if layer.is_incremental() {
                        layer.get_key_range().start
                    } else {
                        layer.get_key_range().end
                    }
                })
                .collect();
            keys.sort();
            Box::new(keys.into_iter())
        };
        let mut keys_iter = keys_iter.peekable();

        // Iter the partition and keys together and query all the necessary
        // keys, computing the max difficulty for each part.
        partitioning
            .parts
            .iter()
            .map(|part| {
                let mut difficulty = 0;
                // Partition ranges are assumed to be sorted and disjoint
                // TODO assert it
                for range in &part.ranges {
                    if !range.is_empty() {
                        difficulty =
                            std::cmp::max(difficulty, self.get_difficulty(lsn, range.start, range));
                    }
                    while let Some(key) = keys_iter.peek() {
                        if key >= &range.end {
                            break;
                        }
                        let key = keys_iter.next().unwrap();
                        if key < range.start {
                            continue;
                        }
                        difficulty =
                            std::cmp::max(difficulty, self.get_difficulty(lsn, key, range));
                    }
                }
                difficulty
            })
            .collect()
    }

    /// For each part of a keyspace partitioning, return the maximum number of layers
    /// that would be needed for page reconstruction in that part at the given LSN.
    ///
    /// If `limit` is provided we don't try to count above that number.
    ///
    /// This method is used to decide where to create new image layers. Computing the
    /// result for the entire partitioning at once allows this function to be more
    /// efficient, and further optimization is possible by using iterators instead,
    /// to allow early return.
    ///
    /// TODO actually use this method instead of count_deltas. Currently we only use
    ///      it for benchmarks.
    pub fn get_difficulty_map(
        &self,
        lsn: Lsn,
        partitioning: &KeyPartitioning,
        limit: Option<usize>,
    ) -> Vec<usize> {
        // TODO This is a naive implementation. Perf improvements to do:
        // 1. Instead of calling self.image_coverage and self.count_deltas,
        //    iterate the image and delta coverage only once.
        partitioning
            .parts
            .iter()
            .map(|part| {
                let mut difficulty = 0;
                for range in &part.ranges {
                    if limit == Some(difficulty) {
                        break;
                    }
                    for (img_range, last_img) in self
                        .image_coverage(range, lsn)
                        .expect("why would this err?")
                    {
                        if limit == Some(difficulty) {
                            break;
                        }
                        let img_lsn = if let Some(last_img) = last_img {
                            last_img.get_lsn_range().end
                        } else {
                            Lsn(0)
                        };

                        if img_lsn < lsn {
                            let num_deltas = self
                                .count_deltas(&img_range, &(img_lsn..lsn), limit)
                                .expect("why would this err lol?");
                            difficulty = std::cmp::max(difficulty, num_deltas);
                        }
                    }
                }
                difficulty
            })
            .collect()
    }

    /// Return all L0 delta layers
    pub fn get_level0_deltas(&self) -> Result<Vec<Arc<PersistentLayerDesc>>> {
        Ok(self.l0_delta_layers.to_vec())
    }

    /// debugging function to print out the contents of the layer map
    #[allow(unused)]
    pub async fn dump(&self, verbose: bool, ctx: &RequestContext) -> Result<()> {
        println!("Begin dump LayerMap");

        println!("open_layer:");
        if let Some(open_layer) = &self.open_layer {
            open_layer.dump(verbose, ctx).await?;
        }

        println!("frozen_layers:");
        for frozen_layer in self.frozen_layers.iter() {
            frozen_layer.dump(verbose, ctx).await?;
        }

        println!("historic_layers:");
        for desc in self.iter_historic_layers() {
            desc.dump();
        }
        println!("End dump LayerMap");
        Ok(())
    }
}
