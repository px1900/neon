//! An ImageLayer represents an image or a snapshot of a key-range at
//! one particular LSN. It contains an image of all key-value pairs
//! in its key-range. Any key that falls into the image layer's range
//! but does not exist in the layer, does not exist.
//!
//! An image layer is stored in a file on disk. The file is stored in
//! timelines/<timeline_id> directory.  Currently, there are no
//! subdirectories, and each image layer file is named like this:
//!
//! ```text
//!    <key start>-<key end>__<LSN>
//! ```
//!
//! For example:
//!
//! ```text
//!    000000067F000032BE0000400000000070B6-000000067F000032BE0000400000000080B6__00000000346BC568
//! ```
//!
//! Every image layer file consists of three parts: "summary",
//! "index", and "values".  The summary is a fixed size header at the
//! beginning of the file, and it contains basic information about the
//! layer, and offsets to the other parts. The "index" is a B-tree,
//! mapping from Key to an offset in the "values" part.  The
//! actual page images are stored in the "values" part.
use crate::config::PageServerConf;
use crate::context::{PageContentKind, RequestContext, RequestContextBuilder};
use crate::page_cache::PAGE_SZ;
use crate::repository::{Key, KEY_SIZE};
use crate::tenant::blob_io::BlobWriter;
use crate::tenant::block_io::{BlockBuf, BlockReader, FileBlockReader};
use crate::tenant::disk_btree::{DiskBtreeBuilder, DiskBtreeReader, VisitDirection};
use crate::tenant::storage_layer::{
    LayerAccessStats, ValueReconstructResult, ValueReconstructState,
};
use crate::tenant::Timeline;
use crate::virtual_file::VirtualFile;
use crate::{IMAGE_FILE_MAGIC, STORAGE_FORMAT_VERSION, TEMP_FILE_SUFFIX};
use anyhow::{bail, ensure, Context, Result};
use bytes::Bytes;
use camino::{Utf8Path, Utf8PathBuf};
use hex;
use pageserver_api::models::LayerAccessKind;
use rand::{distributions::Alphanumeric, Rng};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::SeekFrom;
use std::ops::Range;
use std::os::unix::prelude::FileExt;
use std::sync::Arc;
use tokio::sync::OnceCell;
use tracing::*;

use utils::{
    bin_ser::BeSer,
    id::{TenantId, TimelineId},
    lsn::Lsn,
};

use super::filename::ImageFileName;
use super::{AsLayerDesc, Layer, PersistentLayerDesc, ResidentLayer};

///
/// Header stored in the beginning of the file
///
/// After this comes the 'values' part, starting on block 1. After that,
/// the 'index' starts at the block indicated by 'index_start_blk'
///
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Summary {
    /// Magic value to identify this as a neon image file. Always IMAGE_FILE_MAGIC.
    magic: u16,
    format_version: u16,

    tenant_id: TenantId,
    timeline_id: TimelineId,
    key_range: Range<Key>,
    lsn: Lsn,

    /// Block number where the 'index' part of the file begins.
    index_start_blk: u32,
    /// Block within the 'index', where the B-tree root page is stored
    index_root_blk: u32,
    // the 'values' part starts after the summary header, on block 1.
}

impl From<&ImageLayer> for Summary {
    fn from(layer: &ImageLayer) -> Self {
        Self::expected(
            layer.desc.tenant_id,
            layer.desc.timeline_id,
            layer.desc.key_range.clone(),
            layer.lsn,
        )
    }
}

impl Summary {
    pub(super) fn expected(
        tenant_id: TenantId,
        timeline_id: TimelineId,
        key_range: Range<Key>,
        lsn: Lsn,
    ) -> Self {
        Self {
            magic: IMAGE_FILE_MAGIC,
            format_version: STORAGE_FORMAT_VERSION,
            tenant_id,
            timeline_id,
            key_range,
            lsn,

            index_start_blk: 0,
            index_root_blk: 0,
        }
    }
}

/// This is used only from `pagectl`. Within pageserver, all layers are
/// [`crate::tenant::storage_layer::Layer`], which can hold an [`ImageLayerInner`].
pub struct ImageLayer {
    path: Utf8PathBuf,
    pub desc: PersistentLayerDesc,
    // This entry contains an image of all pages as of this LSN, should be the same as desc.lsn
    pub lsn: Lsn,
    access_stats: LayerAccessStats,
    inner: OnceCell<ImageLayerInner>,
}

impl std::fmt::Debug for ImageLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use super::RangeDisplayDebug;

        f.debug_struct("ImageLayer")
            .field("key_range", &RangeDisplayDebug(&self.desc.key_range))
            .field("file_size", &self.desc.file_size)
            .field("lsn", &self.lsn)
            .field("inner", &self.inner)
            .finish()
    }
}

/// ImageLayer is the in-memory data structure associated with an on-disk image
/// file.
pub struct ImageLayerInner {
    // values copied from summary
    index_start_blk: u32,
    index_root_blk: u32,

    lsn: Lsn,

    /// Reader object for reading blocks from the file.
    file: FileBlockReader,
}

impl std::fmt::Debug for ImageLayerInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageLayerInner")
            .field("index_start_blk", &self.index_start_blk)
            .field("index_root_blk", &self.index_root_blk)
            .finish()
    }
}

impl ImageLayerInner {
    pub(super) async fn dump(&self, ctx: &RequestContext) -> anyhow::Result<()> {
        let file = &self.file;
        let tree_reader =
            DiskBtreeReader::<_, KEY_SIZE>::new(self.index_start_blk, self.index_root_blk, file);

        tree_reader.dump().await?;

        tree_reader
            .visit(
                &[0u8; KEY_SIZE],
                VisitDirection::Forwards,
                |key, value| {
                    println!("key: {} offset {}", hex::encode(key), value);
                    true
                },
                ctx,
            )
            .await?;

        Ok(())
    }
}

/// Boilerplate to implement the Layer trait, always use layer_desc for persistent layers.
impl std::fmt::Display for ImageLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.layer_desc().short_id())
    }
}

impl AsLayerDesc for ImageLayer {
    fn layer_desc(&self) -> &PersistentLayerDesc {
        &self.desc
    }
}

impl ImageLayer {
    pub(crate) async fn dump(&self, verbose: bool, ctx: &RequestContext) -> Result<()> {
        self.desc.dump();

        if !verbose {
            return Ok(());
        }

        let inner = self.load(LayerAccessKind::Dump, ctx).await?;

        inner.dump(ctx).await?;

        Ok(())
    }

    /// Create a temporary file name for a image file.
    fn temp_path_for(
        conf: &PageServerConf,
        timeline_id: TimelineId,
        tenant_id: TenantId,
        fname: &ImageFileName,
    ) -> Utf8PathBuf {
        let rand_string: String = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(8)
            .map(char::from)
            .collect();

        conf.timeline_path(&tenant_id, &timeline_id)
            .join(format!("{fname}.{rand_string}.{TEMP_FILE_SUFFIX}"))
    }

    ///
    /// Open the underlying file and read the metadata into memory, if it's
    /// not loaded already.
    ///
    async fn load(
        &self,
        access_kind: LayerAccessKind,
        ctx: &RequestContext,
    ) -> Result<&ImageLayerInner> {
        self.access_stats.record_access(access_kind, ctx);
        self.inner
            .get_or_try_init(|| self.load_inner(ctx))
            .await
            .with_context(|| format!("Failed to load image layer {}", self.path()))
    }

    /// The main work is ImageLayerInner::load, which is called by this function.
    async fn load_inner(&self, ctx: &RequestContext) -> Result<ImageLayerInner> {
        let path = self.path();

        let loaded = ImageLayerInner::load(&path, self.desc.image_layer_lsn(), None, ctx).await?;

        // not production code
        let actual_filename = path.file_name().unwrap().to_owned();
        let expected_filename = self.layer_desc().filename().file_name();

        if actual_filename != expected_filename {
            println!("warning: filename does not match what is expected from in-file summary");
            println!("actual: {:?}", actual_filename);
            println!("expected: {:?}", expected_filename);
        }

        Ok(loaded)
    }

    /// Create an ImageLayer struct representing an existing file on disk.
    ///
    /// This variant is only used for debugging purposes, by the 'pagectl' binary.
    pub fn new_for_path(path: &Utf8Path, file: File) -> Result<ImageLayer> {
        let mut summary_buf = vec![0; PAGE_SZ];
        file.read_exact_at(&mut summary_buf, 0)?;
        let summary = Summary::des_prefix(&summary_buf)?;
        let metadata = file
            .metadata()
            .context("get file metadata to determine size")?;
        Ok(ImageLayer {
            path: path.to_path_buf(),
            desc: PersistentLayerDesc::new_img(
                summary.tenant_id,
                summary.timeline_id,
                summary.key_range,
                summary.lsn,
                metadata.len(),
            ), // Now we assume image layer ALWAYS covers the full range. This may change in the future.
            lsn: summary.lsn,
            access_stats: LayerAccessStats::empty_will_record_residence_event_later(),
            inner: OnceCell::new(),
        })
    }

    fn path(&self) -> Utf8PathBuf {
        self.path.clone()
    }
}

impl ImageLayerInner {
    // Load an image layer from disk.
    // 1. Open a file using the given path.
    // 2. Read the summary block.
    // 3. Verify the summary block matches the expected summary.
    // 4. Return the file and the summary block.
    // Note that parameter Lsn is only used to be wrapped in the return result.
    pub(super) async fn load(
        path: &Utf8Path,
        lsn: Lsn,
        summary: Option<Summary>,
        ctx: &RequestContext,
    ) -> anyhow::Result<Self> {
        let file = VirtualFile::open(path)
            .await
            .with_context(|| format!("Failed to open file '{}'", path))?;
        let file = FileBlockReader::new(file);
        let summary_blk = file.read_blk(0, ctx).await?;
        let actual_summary = Summary::des_prefix(summary_blk.as_ref())?;

        if let Some(mut expected_summary) = summary {
            // production code path
            expected_summary.index_start_blk = actual_summary.index_start_blk;
            expected_summary.index_root_blk = actual_summary.index_root_blk;

            if actual_summary != expected_summary {
                bail!(
                    "in-file summary does not match expected summary. actual = {:?} expected = {:?}",
                    actual_summary,
                    expected_summary
                );
            }
        }

        Ok(ImageLayerInner {
            index_start_blk: actual_summary.index_start_blk,
            index_root_blk: actual_summary.index_root_blk,
            lsn,
            file,
        })
    }

    /// Fetch a page from file.
    /// First, check the B-Tree index to find the offset of the page.
    /// Then, read the page using the offset.
    pub(super) async fn get_value_reconstruct_data(
        &self,
        key: Key,
        reconstruct_state: &mut ValueReconstructState,
        ctx: &RequestContext,
    ) -> anyhow::Result<ValueReconstructResult> {
        let file = &self.file;
        let tree_reader = DiskBtreeReader::new(self.index_start_blk, self.index_root_blk, file);

        let mut keybuf: [u8; KEY_SIZE] = [0u8; KEY_SIZE];
        key.write_to_byte_slice(&mut keybuf);
        if let Some(offset) = tree_reader
            .get(
                &keybuf,
                &RequestContextBuilder::extend(ctx)
                    .page_content_kind(PageContentKind::ImageLayerBtreeNode)
                    .build(),
            )
            .await?
        {
            let blob = file
                .block_cursor()
                .read_blob(
                    offset,
                    &RequestContextBuilder::extend(ctx)
                        .page_content_kind(PageContentKind::ImageLayerValue)
                        .build(),
                )
                .await
                .with_context(|| format!("failed to read value from offset {}", offset))?;
            let value = Bytes::from(blob);

            reconstruct_state.img = Some((self.lsn, value));
            Ok(ValueReconstructResult::Complete)
        } else {
            Ok(ValueReconstructResult::Missing)
        }
    }
}

/// A builder object for constructing a new image layer.
///
/// Usage:
///
/// 1. Create the ImageLayerWriter by calling ImageLayerWriter::new(...)
///
/// 2. Write the contents by calling `put_page_image` for every key-value
///    pair in the key range.
///
/// 3. Call `finish`.
///
struct ImageLayerWriterInner {
    conf: &'static PageServerConf,
    path: Utf8PathBuf,
    timeline_id: TimelineId,
    tenant_id: TenantId,
    key_range: Range<Key>,
    lsn: Lsn,

    blob_writer: BlobWriter<false>,
    tree: DiskBtreeBuilder<BlockBuf, KEY_SIZE>,
}

/// XI Note;
/// This function works to initialize a new image layer file and file writer.
///
/// 1. Create a new file with a temporary name.
/// 2. Create a new blob writer.
/// 3. Create a new B-Tree writer.
/// 4. Initialize the ImageLayerWriterInner struct using the above objects.
///
impl ImageLayerWriterInner {
    ///
    /// Start building a new image layer.
    ///
    async fn new(
        conf: &'static PageServerConf,
        timeline_id: TimelineId,
        tenant_id: TenantId,
        key_range: &Range<Key>,
        lsn: Lsn,
    ) -> anyhow::Result<Self> {
        // Create the file initially with a temporary filename.
        // We'll atomically rename it to the final name when we're done.
        let path = ImageLayer::temp_path_for(
            conf,
            timeline_id,
            tenant_id,
            &ImageFileName {
                key_range: key_range.clone(),
                lsn,
            },
        );
        info!("new image layer {path}");
        let mut file = VirtualFile::open_with_options(
            &path,
            std::fs::OpenOptions::new().write(true).create_new(true),
        )
        .await?;
        // make room for the header block
        file.seek(SeekFrom::Start(PAGE_SZ as u64)).await?;
        let blob_writer = BlobWriter::new(file, PAGE_SZ as u64);

        // Initialize the b-tree index builder
        let block_buf = BlockBuf::new();
        let tree_builder = DiskBtreeBuilder::new(block_buf);

        let writer = Self {
            conf,
            path,
            timeline_id,
            tenant_id,
            key_range: key_range.clone(),
            lsn,
            tree: tree_builder,
            blob_writer,
        };

        Ok(writer)
    }

    ///
    /// Write next value to the file.
    ///
    /// The page versions must be appended in blknum order.
    ///
    /// 1. Append the image to the blob file.
    /// 2. Append this offset to a B-Tree node corresponding to the key.
    async fn put_image(&mut self, key: Key, img: &[u8]) -> anyhow::Result<()> {
        ensure!(self.key_range.contains(&key));
        let off = self.blob_writer.write_blob(img).await?;

        let mut keybuf: [u8; KEY_SIZE] = [0u8; KEY_SIZE];
        key.write_to_byte_slice(&mut keybuf);
        self.tree.append(&keybuf, off)?;

        Ok(())
    }

    ///
    /// Finish writing the image layer.
    ///
    /// XI Note:
    /// The image layer data arrangement is as follows:
    /// [header BLK - 1 BLK] [Images BLKS] [Index BLKS]
    ///
    /// 1. Write the index to the file
    /// 2. Write the header to the file
    /// 3. Generate a PersistentImageLayer object to call Layer::finish_creating functions
    ///     by this function, it will rename the current temp_path file to the final path (name).
    async fn finish(self, timeline: &Arc<Timeline>) -> anyhow::Result<ResidentLayer> {
        // If blob size is 16K exactly. Then the index_start_blk = 2.999 /1 = 2. Which means start from third page.
        // If blob size is 16K + 1. Then the index_start_blk = 3.000 /1 = 3. Which means start from fourth page.
        // If blob size is 24K - 1. Then the index_start_blk = 3.999 /1 = 3. Which means start from fourth page.
        // So the index_start_blk is always the next page after the last page of the blob.
        let index_start_blk =
            ((self.blob_writer.size() + PAGE_SZ as u64 - 1) / PAGE_SZ as u64) as u32;

        let mut file = self.blob_writer.into_inner();

        // Write out the index
        file.seek(SeekFrom::Start(index_start_blk as u64 * PAGE_SZ as u64))
            .await?;
        // XI Note: By annotation, the tree.finish will flush everything to the disk.
        // However it seems the tree.finish only write the tree into block_buf.
        // And the index_root_blk seems to be an internal root location of the tree,
        // instead of the block number of the blob file.
        let (index_root_blk, block_buf) = self.tree.finish()?;
        for buf in block_buf.blocks {
            file.write_all(buf.as_ref()).await?;
        }

        // Fill in the summary on blk 0
        let summary = Summary {
            magic: IMAGE_FILE_MAGIC,
            format_version: STORAGE_FORMAT_VERSION,
            tenant_id: self.tenant_id,
            timeline_id: self.timeline_id,
            key_range: self.key_range.clone(),
            lsn: self.lsn,
            index_start_blk,
            index_root_blk,
        };

        let mut buf = smallvec::SmallVec::<[u8; PAGE_SZ]>::new();
        Summary::ser_into(&summary, &mut buf)?;
        if buf.spilled() {
            // This is bad as we only have one free block for the summary
            warn!(
                "Used more than one page size for summary buffer: {}",
                buf.len()
            );
        }
        file.seek(SeekFrom::Start(0)).await?;
        file.write_all(&buf).await?;

        let metadata = file
            .metadata()
            .await
            .context("get metadata to determine file size")?;

        let desc = PersistentLayerDesc::new_img(
            self.tenant_id,
            self.timeline_id,
            self.key_range.clone(),
            self.lsn,
            metadata.len(),
        );

        // Note: Because we open the file in write-only mode, we cannot
        // reuse the same VirtualFile for reading later. That's why we don't
        // set inner.file here. The first read will have to re-open it.

        // fsync the file
        file.sync_all().await?;

        // FIXME: why not carry the virtualfile here, it supports renaming?
        let layer = Layer::finish_creating(self.conf, timeline, desc, &self.path)?;

        trace!("created image layer {}", layer.local_path());

        Ok(layer)
    }
}

/// A builder object for constructing a new image layer.
///
/// Usage:
///
/// 1. Create the ImageLayerWriter by calling ImageLayerWriter::new(...)
///
/// 2. Write the contents by calling `put_page_image` for every key-value
///    pair in the key range.
///
/// 3. Call `finish`.
///
/// # Note
///
/// As described in <https://github.com/neondatabase/neon/issues/2650>, it's
/// possible for the writer to drop before `finish` is actually called. So this
/// could lead to odd temporary files in the directory, exhausting file system.
/// This structure wraps `ImageLayerWriterInner` and also contains `Drop`
/// implementation that cleans up the temporary file in failure. It's not
/// possible to do this directly in `ImageLayerWriterInner` since `finish` moves
/// out some fields, making it impossible to implement `Drop`.
///
/// XI Note:
/// Wrapper for ImageLayerWriterInner.
/// Add a Drop implementation to clean up the temporary file in failure.
///
#[must_use]
pub struct ImageLayerWriter {
    inner: Option<ImageLayerWriterInner>,
}

impl ImageLayerWriter {
    ///
    /// Start building a new image layer.
    ///
    pub async fn new(
        conf: &'static PageServerConf,
        timeline_id: TimelineId,
        tenant_id: TenantId,
        key_range: &Range<Key>,
        lsn: Lsn,
    ) -> anyhow::Result<ImageLayerWriter> {
        Ok(Self {
            inner: Some(
                ImageLayerWriterInner::new(conf, timeline_id, tenant_id, key_range, lsn).await?,
            ),
        })
    }

    ///
    /// Write next value to the file.
    ///
    /// The page versions must be appended in blknum order.
    ///
    pub async fn put_image(&mut self, key: Key, img: &[u8]) -> anyhow::Result<()> {
        self.inner.as_mut().unwrap().put_image(key, img).await
    }

    ///
    /// Finish writing the image layer.
    ///
    pub(crate) async fn finish(
        mut self,
        timeline: &Arc<Timeline>,
    ) -> anyhow::Result<super::ResidentLayer> {
        self.inner.take().unwrap().finish(timeline).await
    }
}

impl Drop for ImageLayerWriter {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            inner.blob_writer.into_inner().remove();
        }
    }
}
