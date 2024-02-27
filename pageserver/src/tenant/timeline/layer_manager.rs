use anyhow::{bail, ensure, Context, Result};
use std::{collections::HashMap, sync::Arc};
use tracing::trace;
use utils::{
    id::{TenantId, TimelineId},
    lsn::{AtomicLsn, Lsn},
};

use crate::{
    config::PageServerConf,
    metrics::TimelineMetrics,
    tenant::{
        layer_map::{BatchedUpdates, LayerMap},
        storage_layer::{
            AsLayerDesc, InMemoryLayer, Layer, PersistentLayerDesc, PersistentLayerKey,
            ResidentLayer,
        },
    },
};

/// Provides semantic APIs to manipulate the layer map.
/// XI: The LayerMap is an index from LSN-Key-Range to Layer Descriptor.
///     The LayerFileManager is a map from Layer Descriptor to Layer (which manages the file operations).
pub(crate) struct LayerManager {
    layer_map: LayerMap,
    layer_fmgr: LayerFileManager<Layer>,
}

impl LayerManager {
    pub(crate) fn create() -> Self {
        Self {
            layer_map: LayerMap::default(),
            layer_fmgr: LayerFileManager::new(),
        }
    }

    //XI: Get the layer from LayerFileManager(HashMap) by a PersistentLayerDesc.
    pub(crate) fn get_from_desc(&self, desc: &PersistentLayerDesc) -> Layer {
        self.layer_fmgr.get_from_desc(desc)
    }

    /// Get an immutable reference to the layer map.
    ///
    /// We expect users only to be able to get an immutable layer map. If users want to make modifications,
    /// they should use the below semantic APIs. This design makes us step closer to immutable storage state.
    pub(crate) fn layer_map(&self) -> &LayerMap {
        &self.layer_map
    }

    /// Called from `load_layer_map`. Initialize the layer manager with:
    /// 1. all on-disk layers
    /// 2. next open layer (with disk disk_consistent_lsn LSN)
    ///
    /// XI: This function works to startup the database.
    ///     During startup, the database will load all the layers from disk, and call this function
    ///     to initialize the layer manager. It will insert each disk layer to layer_map (maps from
    ///     LSN-Key-Range to Layer Descriptor) and file manager (maps from Layer Descriptor to Layer).
    ///     It will also set the next open layer at the disk_consistent_lsn, which is used to create
    ///     the first open layer after starting up.
    pub(crate) fn initialize_local_layers(
        &mut self,
        on_disk_layers: Vec<Layer>,
        next_open_layer_at: Lsn,
    ) {
        //XI: updates is a vector to temporarily cache the updates and batch flush them when it was dropped.
        let mut updates = self.layer_map.batch_update();

        //XI: Insert each layer into the layer map(index from LSN-Key-Range to Layer Descriptor) and
        //    file manager (map from Layer Descriptor to Layer).
        for layer in on_disk_layers {
            Self::insert_historic_layer(layer, &mut updates, &mut self.layer_fmgr);
        }
        //XI: Indeed do nothing.
        updates.flush();
        //XI: It helps to create the first open layer after starting up.
        self.layer_map.next_open_layer_at = Some(next_open_layer_at);
    }

    /// Initialize when creating a new timeline, called in `init_empty_layer_map`.
    /// XI: This function works to startup the database without any existing layer (data?).
    pub(crate) fn initialize_empty(&mut self, next_open_layer_at: Lsn) {
        self.layer_map.next_open_layer_at = Some(next_open_layer_at);
    }

    /// Open a new writable layer to append data if there is no open layer, otherwise return the current open layer,
    /// called within `get_layer_for_write`.
    ///
    /// XI: The last_record_lsn is only used to check that current lsn is greater than the last_record_lsn.
    ///     This function also checked when there is an existing open layer, the open layer's start lsn should
    ///     be smaller than the incoming lsn.
    pub(crate) async fn get_layer_for_write(
        &mut self,
        lsn: Lsn,
        last_record_lsn: Lsn,
        conf: &'static PageServerConf,
        timeline_id: TimelineId,
        tenant_id: TenantId,
    ) -> Result<Arc<InMemoryLayer>> {
        ensure!(lsn.is_aligned());

        ensure!(
            lsn > last_record_lsn,
            "cannot modify relation after advancing last_record_lsn (incoming_lsn={}, last_record_lsn={})",
            lsn,
            last_record_lsn,
        );

        // Do we have a layer open for writing already?
        let layer = if let Some(open_layer) = &self.layer_map.open_layer {
            if open_layer.get_lsn_range().start > lsn {
                bail!(
                    "unexpected open layer in the future: open layers starts at {}, write lsn {}",
                    open_layer.get_lsn_range().start,
                    lsn
                );
            }

            Arc::clone(open_layer)
        } else {
            // No writeable layer yet. Create one.
            let start_lsn = self
                .layer_map
                .next_open_layer_at
                .context("No next open layer found")?;

            trace!(
                "creating in-memory layer at {}/{} for record at {}",
                timeline_id,
                start_lsn,
                lsn
            );

            let new_layer = InMemoryLayer::create(conf, timeline_id, tenant_id, start_lsn).await?;
            let layer = Arc::new(new_layer);

            self.layer_map.open_layer = Some(layer.clone());
            self.layer_map.next_open_layer_at = None;

            layer
        };

        Ok(layer)
    }

    /// Called from `freeze_inmem_layer`, returns true if successfully frozen.
    /// XI: 1. Set the end_lsn to the open layer, and do some correctness check. -> open_layer.freeze()
    ///     2. Append this open layer to the frozen_layers vector.
    ///     3. Set the next_open_layer_at to the end_lsn.
    ///     4. Set the return value $last_freeze_at to the end_lsn.
    pub(crate) async fn try_freeze_in_memory_layer(
        &mut self,
        Lsn(last_record_lsn): Lsn,
        last_freeze_at: &AtomicLsn,
    ) {
        let end_lsn = Lsn(last_record_lsn + 1);

        if let Some(open_layer) = &self.layer_map.open_layer {
            let open_layer_rc = Arc::clone(open_layer);
            // Does this layer need freezing?
            // XI: Just set the end_lsn to the open layer, and do some correctness check.
            open_layer.freeze(end_lsn).await;

            // The layer is no longer open, update the layer map to reflect this.
            // We will replace it with on-disk historics below.
            self.layer_map.frozen_layers.push_back(open_layer_rc);
            self.layer_map.open_layer = None;
            self.layer_map.next_open_layer_at = Some(end_lsn);
            last_freeze_at.store(end_lsn);
        }
    }

    /// Add image layers to the layer map, called from `create_image_layers`.
    /// XI: Insert the new image layers to the layer map and file manager.
    ///     Also record the new layers' metrics.
    pub(crate) fn track_new_image_layers(
        &mut self,
        image_layers: &[ResidentLayer],
        metrics: &TimelineMetrics,
    ) {
        let mut updates = self.layer_map.batch_update();
        for layer in image_layers {
            Self::insert_historic_layer(layer.as_ref().clone(), &mut updates, &mut self.layer_fmgr);

            // record these here instead of Layer::finish_creating because otherwise partial
            // failure with create_image_layers would balloon up the physical size gauge. downside
            // is that all layers need to be created before metrics are updated.
            metrics.record_new_file_metrics(layer.layer_desc().file_size);
        }
        updates.flush();
    }

    /// Flush a frozen layer and add the written delta layer to the layer map.
    ///
    /// XI: 1. Get the first(oldest) inmem layer from the frozen_layers vector.
    ///     2. Insert the parameter delta_layer to the layer map and file manager.
    ///     Question: Where does the caller function get the delta_layer?
    ///               It seems we just simply discard the inmem layer.
    ///               I guess the caller function initialize the delta_layer with the inmem layer
    ///               before calling this function.
    pub(crate) fn finish_flush_l0_layer(
        &mut self,
        delta_layer: Option<&ResidentLayer>,
        frozen_layer_for_check: &Arc<InMemoryLayer>,
        metrics: &TimelineMetrics,
    ) {
        //XI: Get the first(oldest) inmem layer from the frozen_layers vector.
        let inmem = self
            .layer_map
            .frozen_layers
            .pop_front()
            .expect("there must be a inmem layer to flush");

        // Only one task may call this function at a time (for this
        // timeline). If two tasks tried to flush the same frozen
        // layer to disk at the same time, that would not work.
        assert_eq!(Arc::as_ptr(&inmem), Arc::as_ptr(frozen_layer_for_check));

        if let Some(l) = delta_layer {
            // XI: The $updates is only a necessary structure to do layer_map update.
            //     updates.flush() will actually apply the updates to the layer map.
            let mut updates = self.layer_map.batch_update();
            Self::insert_historic_layer(l.as_ref().clone(), &mut updates, &mut self.layer_fmgr);
            metrics.record_new_file_metrics(l.layer_desc().file_size);
            updates.flush();
        }
    }

    /// Called when compaction is completed.
    ///
    /// XI: 1. Insert the compact_to layers to the layer map and file manager.
    ///     2. Remove the compact_from layers from the layer map and file manager.
    ///     This may cause rebuilding the layer map index.
    pub(crate) fn finish_compact_l0(
        &mut self,
        layer_removal_cs: &Arc<tokio::sync::OwnedMutexGuard<()>>,
        compact_from: &[Layer],
        compact_to: &[ResidentLayer],
        metrics: &TimelineMetrics,
    ) {
        let mut updates = self.layer_map.batch_update();
        for l in compact_to {
            Self::insert_historic_layer(l.as_ref().clone(), &mut updates, &mut self.layer_fmgr);
            metrics.record_new_file_metrics(l.layer_desc().file_size);
        }
        for l in compact_from {
            Self::delete_historic_layer(layer_removal_cs, l, &mut updates, &mut self.layer_fmgr);
        }
        updates.flush();
    }

    /// Called when garbage collect the timeline. Returns a guard that will apply the updates to the layer map.
    ///
    /// XI: Delete the doomed layers from the layer map and file manager.
    pub(crate) fn finish_gc_timeline(
        &mut self,
        layer_removal_cs: &Arc<tokio::sync::OwnedMutexGuard<()>>,
        gc_layers: Vec<Layer>,
    ) {
        let mut updates = self.layer_map.batch_update();
        for doomed_layer in gc_layers {
            Self::delete_historic_layer(
                layer_removal_cs,
                &doomed_layer,
                &mut updates,
                &mut self.layer_fmgr,
            );
        }
        updates.flush()
    }

    /// Helper function to insert a layer into the layer map and file manager.
    ///
    /// XI: $updates is the layer_map index (maps from LSN-Key-Range to Layer Descriptor).
    ///     $mapping is the file manager (maps from Layer Descriptor to Layer).
    ///     $updates is the necessary structure to do layer_map update.
    ///     updates.flush() will actually apply the updates to the layer map. (auto called by Drop())
    fn insert_historic_layer(
        layer: Layer,
        updates: &mut BatchedUpdates<'_>,
        mapping: &mut LayerFileManager<Layer>,
    ) {
        updates.insert_historic(layer.layer_desc().clone());
        mapping.insert(layer);
    }

    /// Removes the layer from local FS (if present) and from memory.
    /// Remote storage is not affected by this operation.
    fn delete_historic_layer(
        // we cannot remove layers otherwise, since gc and compaction will race
        _layer_removal_cs: &Arc<tokio::sync::OwnedMutexGuard<()>>,
        layer: &Layer,
        updates: &mut BatchedUpdates<'_>,
        mapping: &mut LayerFileManager<Layer>,
    ) {
        let desc = layer.layer_desc();

        // TODO Removing from the bottom of the layer map is expensive.
        //      Maybe instead discard all layer map historic versions that
        //      won't be needed for page reconstruction for this timeline,
        //      and mark what we can't delete yet as deleted from the layer
        //      map index without actually rebuilding the index.
        updates.remove_historic(desc);
        mapping.remove(layer);
        layer.garbage_collect_on_drop();
    }

    pub(crate) fn contains(&self, layer: &Layer) -> bool {
        self.layer_fmgr.contains(layer)
    }
}

//XI: The default T is Layer, so the LayerFileManager is a map from PersistentLayerKey to Layer.
pub(crate) struct LayerFileManager<T>(HashMap<PersistentLayerKey, T>);

impl<T: AsLayerDesc + Clone> LayerFileManager<T> {
    // XI: Get a layer from the hashmap by a PersistentLayerDesc.
    fn get_from_desc(&self, desc: &PersistentLayerDesc) -> T {
        // The assumption for the `expect()` is that all code maintains the following invariant:
        // A layer's descriptor is present in the LayerMap => the LayerFileManager contains a layer for the descriptor.
        self.0
            .get(&desc.key())
            .with_context(|| format!("get layer from desc: {}", desc.filename()))
            .expect("not found")
            .clone()
    }

    //XI: Overwrite the layer with the same key.
    pub(crate) fn insert(&mut self, layer: T) {
        let present = self.0.insert(layer.layer_desc().key(), layer.clone());
        if present.is_some() && cfg!(debug_assertions) {
            panic!("overwriting a layer: {:?}", layer.layer_desc())
        }
    }

    pub(crate) fn contains(&self, layer: &T) -> bool {
        self.0.contains_key(&layer.layer_desc().key())
    }

    pub(crate) fn new() -> Self {
        Self(HashMap::new())
    }

    pub(crate) fn remove(&mut self, layer: &T) {
        let present = self.0.remove(&layer.layer_desc().key());
        if present.is_none() && cfg!(debug_assertions) {
            panic!(
                "removing layer that is not present in layer mapping: {:?}",
                layer.layer_desc()
            )
        }
    }
}
