use std::ops::Range;

// NOTE the `im` crate has 20x more downloads and also has
// persistent/immutable BTree. But it's bugged so rpds is a
// better choice <https://github.com/neondatabase/neon/issues/3395>
use rpds::RedBlackTreeMapSync;

/// Data structure that can efficiently:
/// - find the latest layer by lsn.end at a given key
/// - iterate the latest layers in a key range
/// - insert layers in non-decreasing lsn.start order
///
/// For a detailed explanation and justification of this approach, see:
/// <https://neon.tech/blog/persistent-structures-in-neons-wal-indexing>
///
/// NOTE The struct is parameterized over Value for easier
///      testing, but in practice it's some sort of layer.

// Key is RelationId, Value is (lsn.end, LayerFile)
// One LayerCoverage is a map for one layer.
pub struct LayerCoverage<Value> {
    /// For every change in coverage (as we sweep the key space)
    /// we store (lsn.end, value).
    ///
    /// NOTE We use an immutable/persistent tree so that we can keep historic
    ///      versions of this coverage without cloning the whole thing and
    ///      incurring quadratic memory cost. See HistoricLayerCoverage.
    ///
    /// NOTE We use the Sync version of the map because we want Self to
    ///      be Sync. Using nonsync might be faster, if we can work with
    ///      that.
    nodes: RedBlackTreeMapSync<i128, Option<(u64, Value)>>,
}

impl<T: Clone> Default for LayerCoverage<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Value: Clone> LayerCoverage<Value> {
    pub fn new() -> Self {
        Self {
            nodes: RedBlackTreeMapSync::default(),
        }
    }

    /// Helper function to subdivide the key range without changing any values
    ///
    /// This operation has no semantic effect by itself. It only helps us pin in
    /// place the part of the coverage we don't want to change when inserting.
    ///
    /// As an analogy, think of a polygon. If you add a vertex along one of the
    /// segments, the polygon is still the same, but it behaves differently when
    /// we move or delete one of the other points.
    ///
    /// Complexity: O(log N)
    fn add_node(&mut self, key: i128) {
        // In fact, the range.last returns the biggest key that is less than or equal to the given key.
        let value = match self.nodes.range(..=key).last() {
            Some((_, Some(v))) => Some(v.clone()),
            Some((_, None)) => None,
            None => None,
        };
        self.nodes.insert_mut(key, value);
    }

    /// Insert a layer.
    ///
    /// Complexity: worst case O(N), in practice O(log N). See NOTE in implementation.
    ///
    /// XI Note:
    /// 1. Add two keys to the tree, and the their value copy the previously key's value.
    ///    After, insertion, the tree is same with the previous tree. The only difference is
    ///    that it has two more redundant keys.
    /// 2. Now, we need to update the lsn's LSN and values in the tree.
    ///    We iterate all the keys in the range [key.start, key.end).
    ///    For the value of each key, if it is None, it means that this key hasn't been recorded
    ///    in the layer, so we need to update it. Similar, if there is a layer exists for this key,
    ///    but this layer is outdated now, which means the existed layer's lsn.end is smaller than
    ///    the new parameter lsn.end, we need to update this key's LSN and value.
    /// 3. For all the keys that need to be updated, we need to update their values and LSNs to the
    ///    latest parameter one. But one exception is that, if the continuously previous key has been
    ///    updated to the newest, and the current key is also required to be updated to newest one,
    ///    we will have two continuous key with same LSN and same value, which is redundant and will
    ///    cause performance drop for searching. So, in this case we will delete the current key from tree.
    pub fn insert(&mut self, key: Range<i128>, lsn: Range<u64>, value: Value) {
        // Add nodes at endpoints
        //
        // NOTE The order of lines is important. We add nodes at the start
        // and end of the key range **before updating any nodes** in order
        // to pin down the current coverage outside of the relevant key range.
        // Only the coverage inside the layer's key range should change.
        self.add_node(key.start);
        self.add_node(key.end);

        // Raise the height where necessary
        //
        // NOTE This loop is worst case O(N), but amortized O(log N) in the special
        // case when rectangles have no height. In practice I don't think we'll see
        // the kind of layer intersections needed to trigger O(N) behavior. The worst
        // case is N/2 horizontal layers overlapped with N/2 vertical layers in a
        // grid pattern.
        let mut to_update = Vec::new();
        let mut to_remove = Vec::new();
        let mut prev_covered = false;
        for (k, node) in self.nodes.range(key) {
            let needs_cover = match node {
                None => true,
                Some((h, _)) => h < &lsn.end,
            };
            if needs_cover {
                match prev_covered {
                    true => to_remove.push(*k),
                    false => to_update.push(*k),
                }
            }
            prev_covered = needs_cover;
        }
        // TODO check if the nodes inserted at key.start and key.end are safe
        //      to remove. It's fine to keep them but they could be redundant.
        for k in to_update {
            self.nodes.insert_mut(k, Some((lsn.end, value.clone())));
        }
        for k in to_remove {
            self.nodes.remove_mut(&k);
        }
    }

    /// Get the latest (by lsn.end) layer at a given key
    ///
    /// Complexity: O(log N)
    ///
    /// Xi Note:
    /// If the tree has five nodes: key_1, key_50, key_100, key_150, key_200
    /// The key_1 manages the key range [1, 50)
    /// The key_50 manages the key range [50, 100) and so on.
    /// So, for example, there is a key_60 coming, the range() will search the range [key_1, key_60)
    /// The range() will return [key_1, key_50],
    ///    then the next_back() will return key_50, which manages the key_60
    pub fn query(&self, key: i128) -> Option<Value> {
        self.nodes
            .range(..=key)
            .next_back()?
            .1
            .as_ref()
            .map(|(_, v)| v.clone())
    }

    /// Iterate the changes in layer coverage in a given range. You will likely
    /// want to start with self.query(key.start), and then follow up with self.range
    ///
    /// Complexity: O(log N + result_size)
    ///
    /// Xi Note:
    /// It seems the value format is a compound and the true value is the second element.
    pub fn range(&self, key: Range<i128>) -> impl '_ + Iterator<Item = (i128, Option<Value>)> {
        self.nodes
            .range(key)
            .map(|(k, v)| (*k, v.as_ref().map(|x| x.1.clone())))
    }

    /// O(1) clone
    pub fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
        }
    }
}

/// Image and delta coverage at a specific LSN.
/// Xi Note:
/// For a specific time (specific status/LSN), one compute node will maps to one LayerCoverageTuple.
/// Inside this tuple, there are two LayerCoverage, one is image_coverage, the other is delta_coverage.
pub struct LayerCoverageTuple<Value> {
    pub image_coverage: LayerCoverage<Value>,
    pub delta_coverage: LayerCoverage<Value>,
}

impl<T: Clone> Default for LayerCoverageTuple<T> {
    fn default() -> Self {
        Self {
            image_coverage: LayerCoverage::default(),
            delta_coverage: LayerCoverage::default(),
        }
    }
}

impl<Value: Clone> LayerCoverageTuple<Value> {
    pub fn clone(&self) -> Self {
        Self {
            image_coverage: self.image_coverage.clone(),
            delta_coverage: self.delta_coverage.clone(),
        }
    }
}
