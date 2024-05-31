import logging
from typing import List

import ctypes
import PyNvVideoCodec as nvc
import nvcv
from av import VideoFrame
from av.frame import Frame

from ..jitterbuffer import JitterFrame
from ..mediastreams import VIDEO_TIME_BASE
from .base import Decoder

logger = logging.getLogger(__name__)

class CudaFrame(VideoFrame):
    """
    Stores a decoded frame from the CUDA decoder.

    It's expected that the decoded frame (which can be accessed either via to_nvcv_tensor,
    or decoded_frame) will undergo further processing on the GPU,
    before being reconstituted into another av.Frame that can be fed to an encoder.
    """

    def __new__(cls, decoded_frame: nvc.DecodedFrame = None, decoded_frame_width: int = 0, decoded_frame_height: int = 0):
        return super().__new__(cls, width=0, height=0, format="yuv420p")

    def __init__(self, decoded_frame: nvc.DecodedFrame = None, decoded_frame_width: int = 0, decoded_frame_height: int = 0):
        # create VideoFrame with zero width and height to prevent allocation
        # of underlying plane buffers
        self.decoded_frame_width = decoded_frame_width
        self.decoded_frame_height = decoded_frame_height
        self.decoded_frame = decoded_frame

    @property
    def width(self):
        """Width of the image, in pixels."""
        return self.decoded_frame_width

    @property
    def height(self):
        """Height of the image, in pixels."""
        return self.decoded_frame_height

    def to_nvcv_tensor(self):
        return nvcv.as_tensor(nvcv.as_image(self.decoded_frame.nvcv_image(), nvcv.Format.U8))


class CudaH264Decoder(Decoder):
    def __init__(self) -> None:
        self.nv_dec = nvc.CreateDecoder(codec=nvc.cudaVideoCodec.H264, usedevicememory=True, enableasyncallocations=False)
        self.packet = nvc.PacketData()
        self.packet_c_bytes = None

    def decode(self, encoded_frame: JitterFrame) -> List[Frame]:
        frames = []
        
        if (self.packet_c_bytes is None) or (len(encoded_frame.data) > len(self.packet_c_bytes)):
            self.packet_c_bytes = (ctypes.c_byte * len(encoded_frame.data)).from_buffer_copy(encoded_frame.data)
        else:
            ctypes.memmove(ctypes.addressof(self.packet_c_bytes), encoded_frame.data, len(encoded_frame.data))
        
        self.packet.bsl_data = ctypes.cast(self.packet_c_bytes, ctypes.c_void_p).value
        self.packet.bsl = len(encoded_frame.data)

        for decoded_frame in self.nv_dec.Decode(self.packet):
            # NOTE: there's no real reason we couldn't support other formats here,
            # main issue is finding a suitable VideoFrame format to match
            assert decoded_frame.format == nvc.Pixel_Format.NV12, "unsupported pixel format"
            frame = CudaFrame(
                decoded_frame,
                decoded_frame_width=self.nv_dec.GetWidth(),
                decoded_frame_height=self.nv_dec.GetHeight()
            )
            # decoded_frame.timestamp always seems to be zero, so use encoded_frame.timestamp instead 
            frame.pts = encoded_frame.timestamp
            frame.time_base = VIDEO_TIME_BASE
            frames.append(frame)

        return frames
