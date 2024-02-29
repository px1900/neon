//!
//! Functions for reading and writing variable-sized "blobs".
//!
//! Each blob begins with a 1- or 4-byte length field, followed by the
//! actual data. If the length is smaller than 128 bytes, the length
//! is written as a one byte. If it's larger than that, the length
//! is written as a four-byte integer, in big-endian, with the high
//! bit set. This way, we can detect whether it's 1- or 4-byte header
//! by peeking at the first byte.
//!
//! len <  128: 0XXXXXXX
//! len >= 128: 1XXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX
//!
use crate::context::RequestContext;
use crate::page_cache::PAGE_SZ;
use crate::tenant::block_io::BlockCursor;
use crate::virtual_file::VirtualFile;
use std::cmp::min;
use std::io::{Error, ErrorKind};

impl<'a> BlockCursor<'a> {
    /// Read a blob into a new buffer.
    pub async fn read_blob(
        &self,
        offset: u64,
        ctx: &RequestContext,
    ) -> Result<Vec<u8>, std::io::Error> {
        let mut buf = Vec::new();
        self.read_blob_into_buf(offset, &mut buf, ctx).await?;
        Ok(buf)
    }
    /// Read blob into the given buffer. Any previous contents in the buffer
    /// are overwritten.
    ///
    /// XI: This function is implemented using block io.
    ///     1. Read the first block and get the length of the blob
    ///     2. Continuously read the data using the block io and extend the buffer with the data
    ///        until it gets the whole blob
    pub async fn read_blob_into_buf(
        &self,
        offset: u64,
        dstbuf: &mut Vec<u8>,
        ctx: &RequestContext,
    ) -> Result<(), std::io::Error> {
        //XI: Calculate the block number and offset inside the block
        let mut blknum = (offset / PAGE_SZ as u64) as u32;
        let mut off = (offset % PAGE_SZ as u64) as usize;

        //XI: Read the first block of this blob
        let mut buf = self.read_blk(blknum, ctx).await?;

        // peek at the first byte, to determine if it's a 1- or 4-byte length
        let first_len_byte = buf[off];

        //XI: Get the length of the blob. It use the first_byte to tell whether the length use 1 or
        //    4 bytes to store
        let len: usize = if first_len_byte < 0x80 {
            // 1-byte length header
            off += 1;
            first_len_byte as usize
        } else {
            // 4-byte length header
            let mut len_buf = [0u8; 4];
            let thislen = PAGE_SZ - off;
            if thislen < 4 {
                // it is split across two pages
                len_buf[..thislen].copy_from_slice(&buf[off..PAGE_SZ]);
                blknum += 1;
                buf = self.read_blk(blknum, ctx).await?;
                len_buf[thislen..].copy_from_slice(&buf[0..4 - thislen]);
                off = 4 - thislen;
            } else {
                len_buf.copy_from_slice(&buf[off..off + 4]);
                off += 4;
            }
            len_buf[0] &= 0x7f;
            u32::from_be_bytes(len_buf) as usize
        };

        //XI: dstbuf is the buffer to store the blob
        dstbuf.clear();
        dstbuf.reserve(len);

        // Read the payload
        // XI: Continuously read current block's data to buf, and then extend the dstbuf with the data in buf
        let mut remain = len;
        while remain > 0 {
            let mut page_remain = PAGE_SZ - off;
            if page_remain == 0 {
                // continue on next page
                blknum += 1;
                buf = self.read_blk(blknum, ctx).await?;
                off = 0;
                page_remain = PAGE_SZ;
            }
            //XI: How many data should be read from this block
            let this_blk_len = min(remain, page_remain);
            dstbuf.extend_from_slice(&buf[off..off + this_blk_len]);
            remain -= this_blk_len;
            off += this_blk_len;
        }
        Ok(())
    }
}

/// A wrapper of `VirtualFile` that allows users to write blobs.
///
/// If a `BlobWriter` is dropped, the internal buffer will be
/// discarded. You need to call [`flush_buffer`](Self::flush_buffer)
/// manually before dropping.
///
/// XI: What is the purpose of offset? Does it represent the currently occupied buffer size?
///     Updates: the offset donates how much data has been written to the file.
///
///    1. It will use write_blob to write a blob to the file
///    2. During the write_blob, it will write the length of the blob and data separately.
///    3. For each writing, it will call the write_all function to write the data.
///    4. The data to be written will be firstly written to the buffer, and then flush the buffer to the file.
///    5. When we try to write some data to blob, firstly we need to check whether there are some
///       data unflushed and cached inside the write->buffer.
pub struct BlobWriter<const BUFFERED: bool> {
    inner: VirtualFile,
    offset: u64,
    /// A buffer to save on write calls, only used if BUFFERED=true
    buf: Vec<u8>,
}

impl<const BUFFERED: bool> BlobWriter<BUFFERED> {
    pub fn new(inner: VirtualFile, start_offset: u64) -> Self {
        Self {
            inner,
            offset: start_offset,
            buf: Vec::with_capacity(Self::CAPACITY),
        }
    }

    pub fn size(&self) -> u64 {
        self.offset
    }

    const CAPACITY: usize = if BUFFERED { PAGE_SZ } else { 0 };

    #[inline(always)]
    /// Writes the given buffer directly to the underlying `VirtualFile`.
    /// You need to make sure that the internal buffer is empty, otherwise
    /// data will be written in wrong order.
    async fn write_all_unbuffered(&mut self, src_buf: &[u8]) -> Result<(), Error> {
        self.inner.write_all(src_buf).await?;
        self.offset += src_buf.len() as u64;
        Ok(())
    }

    #[inline(always)]
    /// Flushes the internal buffer to the underlying `VirtualFile`.
    pub async fn flush_buffer(&mut self) -> Result<(), Error> {
        self.inner.write_all(&self.buf).await?;
        self.buf.clear();
        Ok(())
    }

    #[inline(always)]
    /// Writes as much of `src_buf` into the internal buffer as it fits
    fn write_into_buffer(&mut self, src_buf: &[u8]) -> usize {
        let remaining = Self::CAPACITY - self.buf.len();
        let to_copy = src_buf.len().min(remaining);
        self.buf.extend_from_slice(&src_buf[..to_copy]);
        self.offset += to_copy as u64;
        to_copy
    }

    /// Internal, possibly buffered, write function
    ///
    /// XI: If BUFFERED feature is disabled, then write the src_buf directly to the file
    ///     If BUFFERED feature is enabled, then:
    ///     1. If there already have some data in the buffer, then to save disk-flushing cost, we
    ///       firstly write as much parameter src_buf to the buffer as it fits, and flush the whole
    ///       buffer once to the file. Then, write the remaining src_buf to the file bypassing buffer.
    ///     2. If the buffer is empty, then write the src_buf directly to the file.
    async fn write_all(&mut self, mut src_buf: &[u8]) -> Result<(), Error> {
        if !BUFFERED {
            assert!(self.buf.is_empty());
            self.write_all_unbuffered(src_buf).await?;
            return Ok(());
        }
        let remaining = Self::CAPACITY - self.buf.len();
        // First try to copy as much as we can into the buffer
        if remaining > 0 {
            let copied = self.write_into_buffer(src_buf);
            src_buf = &src_buf[copied..];
        }
        // Then, if the buffer is full, flush it out
        if self.buf.len() == Self::CAPACITY {
            self.flush_buffer().await?;
        }
        // Finally, write the tail of src_buf:
        // If it wholly fits into the buffer without
        // completely filling it, then put it there.
        // If not, write it out directly.
        if !src_buf.is_empty() {
            assert_eq!(self.buf.len(), 0);
            if src_buf.len() < Self::CAPACITY {
                let copied = self.write_into_buffer(src_buf);
                // We just verified above that src_buf fits into our internal buffer.
                assert_eq!(copied, src_buf.len());
            } else {
                self.write_all_unbuffered(src_buf).await?;
            }
        }
        Ok(())
    }

    /// Write a blob of data. Returns the offset that it was written to,
    /// which can be used to retrieve the data later.
    ///
    /// XI: This function works to write a blob to the file.
    ///     Firstly, it write the length of the blob to the file
    ///     Then, it write the blob data to the file
    pub async fn write_blob(&mut self, srcbuf: &[u8]) -> Result<u64, Error> {
        let offset = self.offset;

        if srcbuf.len() < 128 {
            // Short blob. Write a 1-byte length header
            let len_buf = srcbuf.len() as u8;
            self.write_all(&[len_buf]).await?;
        } else {
            // Write a 4-byte length header
            if srcbuf.len() > 0x7fff_ffff {
                return Err(Error::new(
                    ErrorKind::Other,
                    format!("blob too large ({} bytes)", srcbuf.len()),
                ));
            }
            let mut len_buf = ((srcbuf.len()) as u32).to_be_bytes();
            len_buf[0] |= 0x80;
            self.write_all(&len_buf).await?;
        }
        self.write_all(srcbuf).await?;
        Ok(offset)
    }
}

//XI: There are two types of BlobWriter, one is with BUFFERED feature, and the other is without BUFFERED feature
//    With BUFFERED, during get the virtual file, it has two approach: flushing the buffer before giving access
//    or giving access without flushing buffer.
//    Without BUFFERED, it only have into_inner, which doesn't involve buffer.
impl BlobWriter<true> {
    /// Access the underlying `VirtualFile`.
    ///
    /// This function flushes the internal buffer before giving access
    /// to the underlying `VirtualFile`.
    pub async fn into_inner(mut self) -> Result<VirtualFile, Error> {
        self.flush_buffer().await?;
        Ok(self.inner)
    }

    /// Access the underlying `VirtualFile`.
    ///
    /// Unlike [`into_inner`](Self::into_inner), this doesn't flush
    /// the internal buffer before giving access.
    pub fn into_inner_no_flush(self) -> VirtualFile {
        self.inner
    }
}

impl BlobWriter<false> {
    /// Access the underlying `VirtualFile`.
    pub fn into_inner(self) -> VirtualFile {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::DownloadBehavior, task_mgr::TaskKind, tenant::block_io::BlockReaderRef};
    use rand::{Rng, SeedableRng};

    async fn round_trip_test<const BUFFERED: bool>(blobs: &[Vec<u8>]) -> Result<(), Error> {
        let temp_dir = camino_tempfile::tempdir()?;
        let pathbuf = temp_dir.path().join("file");
        let ctx = RequestContext::new(TaskKind::UnitTest, DownloadBehavior::Error);

        // Write part (in block to drop the file)
        let mut offsets = Vec::new();
        {
            let file = VirtualFile::create(pathbuf.as_path()).await?;
            let mut wtr = BlobWriter::<BUFFERED>::new(file, 0);
            for blob in blobs.iter() {
                let offs = wtr.write_blob(blob).await?;
                offsets.push(offs);
            }
            // Write out one page worth of zeros so that we can
            // read again with read_blk
            let offs = wtr.write_blob(&vec![0; PAGE_SZ]).await?;
            println!("Writing final blob at offs={offs}");
            wtr.flush_buffer().await?;
        }

        let file = VirtualFile::open(pathbuf.as_path()).await?;
        let rdr = BlockReaderRef::VirtualFile(&file);
        let rdr = BlockCursor::new(rdr);
        for (idx, (blob, offset)) in blobs.iter().zip(offsets.iter()).enumerate() {
            let blob_read = rdr.read_blob(*offset, &ctx).await?;
            assert_eq!(
                blob, &blob_read,
                "mismatch for idx={idx} at offset={offset}"
            );
        }
        Ok(())
    }

    fn random_array(len: usize) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        (0..len).map(|_| rng.gen()).collect::<_>()
    }

    #[tokio::test]
    async fn test_one() -> Result<(), Error> {
        let blobs = &[vec![12, 21, 22]];
        round_trip_test::<false>(blobs).await?;
        round_trip_test::<true>(blobs).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_hello_simple() -> Result<(), Error> {
        let blobs = &[
            vec![0, 1, 2, 3],
            b"Hello, World!".to_vec(),
            Vec::new(),
            b"foobar".to_vec(),
        ];
        round_trip_test::<false>(blobs).await?;
        round_trip_test::<true>(blobs).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_really_big_array() -> Result<(), Error> {
        let blobs = &[
            b"test".to_vec(),
            random_array(10 * PAGE_SZ),
            b"foobar".to_vec(),
        ];
        round_trip_test::<false>(blobs).await?;
        round_trip_test::<true>(blobs).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_arrays_inc() -> Result<(), Error> {
        let blobs = (0..PAGE_SZ / 8)
            .map(|v| random_array(v * 16))
            .collect::<Vec<_>>();
        round_trip_test::<false>(&blobs).await?;
        round_trip_test::<true>(&blobs).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_arrays_random_size() -> Result<(), Error> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let blobs = (0..1024)
            .map(|_| {
                let mut sz: u16 = rng.gen();
                // Make 50% of the arrays small
                if rng.gen() {
                    sz |= 63;
                }
                random_array(sz.into())
            })
            .collect::<Vec<_>>();
        round_trip_test::<false>(&blobs).await?;
        round_trip_test::<true>(&blobs).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_arrays_page_boundary() -> Result<(), Error> {
        let blobs = &[
            random_array(PAGE_SZ - 4),
            random_array(PAGE_SZ - 4),
            random_array(PAGE_SZ - 4),
        ];
        round_trip_test::<false>(blobs).await?;
        round_trip_test::<true>(blobs).await?;
        Ok(())
    }
}
