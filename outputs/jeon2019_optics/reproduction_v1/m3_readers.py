"""M3真实格式读取：MAT-v5流式色谱块、HDF5切片及EXR扫描行。"""
import io
import re
import struct
import sys
import zlib
from pathlib import Path
import numpy as np
from scipy.io import whosmat,loadmat
from audit import HERE

DEPENDENCIES=HERE/"dependencies/m3_readers"
if DEPENDENCIES.exists():sys.path.insert(0,str(DEPENDENCIES))
WAVES=np.arange(420,661,10)
DTYPES={1:"i1",2:"u1",3:"<i2",4:"<u2",5:"<i4",6:"<u4",7:"<f4",9:"<f8",12:"<i8",13:"<u8"}


def wavelength_indices(values):
    wave=np.asarray(values,dtype=float).reshape(-1)
    if not np.isfinite(wave).all() or len(np.unique(wave))!=len(wave):raise ValueError("波长重复或非有限")
    indices=[]
    for target in WAVES:
        found=np.flatnonzero(abs(wave-target)<.01)
        if len(found)!=1:raise ValueError(f"缺少或重复{target}nm，禁止补零")
        indices.append(int(found[0]))
    return indices


def exr_wavelength_names(names):
    mapping={}
    for name in names:
        match=re.fullmatch(r"w(\d+(?:\.\d+)?)nm",name)
        if match:
            value=float(match.group(1))
            if value in mapping:raise ValueError("EXR重复波长名称")
            mapping[value]=name
    return [mapping[float(w)] for w in WAVES] if set(WAVES)<=set(mapping) else (_ for _ in ()).throw(ValueError("EXR实际通道不含w420nm至w660nm；不按数组序号猜测"))


class InflateReader:
    """按需解压单个miCOMPRESSED元素，输出缓冲不超过调用读取量加64KiB。"""
    def __init__(self,stream,size):self.stream=stream;self.remaining=size;self.z=zlib.decompressobj();self.buffer=b"";self.tail=b""
    def read(self,n):
        while len(self.buffer)<n:
            if not self.tail:
                if not self.remaining:break
                self.tail=self.stream.read(min(65536,self.remaining));self.remaining-=len(self.tail)
                if not self.tail:raise ValueError("MAT压缩元素提前EOF")
            self.buffer+=self.z.decompress(self.tail,max(65536,n-len(self.buffer)))
            self.tail=self.z.unconsumed_tail
            if self.z.eof and len(self.buffer)<n:break
        result=self.buffer[:n];self.buffer=self.buffer[n:]
        return result


def exact(stream,count):
    value=stream.read(count)
    if len(value)!=count:raise ValueError("MAT数据元素截短")
    return value


def tag(stream):
    a,b=struct.unpack("<II",exact(stream,8))
    if a>>16:return a&65535,a>>16,struct.pack("<I",b)[:a>>16],True
    return a,b,None,False


def element(stream):
    kind,size,small,is_small=tag(stream)
    if is_small:return kind,small
    value=exact(stream,size)
    if size%8:exact(stream,8-size%8)
    return kind,value


def matlab_ref_blocks(path,selected,columns=64):
    """按波段与列块读取MAT-v5双精度ref，不分配完整三维场景。"""
    with Path(path).open("rb") as raw:
        header=exact(raw,128)
        if header[126:128]!=b"IM":raise ValueError("只接受已审核的小端MAT-v5")
        while True:
            block=raw.read(8)
            if not block:break
            if len(block)!=8:raise ValueError("MAT顶层标签截短")
            kind,size=struct.unpack("<II",block);end=raw.tell()+size
            stream=InflateReader(raw,size) if kind==15 else raw
            if kind==15:
                inner,inner_size,_,small=tag(stream)
                if inner!=14 or small:raise ValueError("MAT压缩内容不是矩阵")
            elif kind!=14:
                raw.seek(end);continue
            flags_kind,flags=element(stream);dims_kind,dims=element(stream);name_kind,name=element(stream)
            dimensions=np.frombuffer(dims,dtype="<i4").tolist()
            if name.rstrip(b"\x00")!=b"ref":raw.seek(end);continue
            flag=int(np.frombuffer(flags,dtype="<u4")[0])
            if flag&0x800:raise ValueError("ref不能是复数")
            dtype,size,small,is_small=tag(stream)
            if is_small or dtype not in DTYPES or len(dimensions)!=3:raise ValueError("ref数据类型/维度不符")
            h,w,bands=dimensions;dt=np.dtype(DTYPES[dtype])
            if size!=h*w*bands*dt.itemsize:raise ValueError("ref元素长度不符")
            for band in range(bands):
                for x in range(0,w,columns):
                    width=min(columns,w-x);values=exact(stream,h*width*dt.itemsize)
                    if band in selected:
                        yield band,x,np.frombuffer(values,dtype=dt).reshape((h,width),order="F")
            if kind==15:
                while stream.read(65536):pass
                if not stream.z.eof:raise ValueError("MAT压缩流校验未完成")
            return
    raise ValueError("MAT未找到ref矩阵")


class Harvard:
    def __init__(self,path,calibration,readme):
        self.path=Path(path);self.calibration=np.loadtxt(calibration,dtype=float).reshape(-1)
        text=Path(readme).read_text(encoding="utf-8",errors="strict")
        if "420:10:720" not in text or "lbl" not in text or "sensitivity" not in text:raise ValueError("Harvard实际README不能证明波长/掩码/校正")
        variables={name:(shape,kind) for name,shape,kind in whosmat(path,appendmat=False)}
        if "ref" not in variables or "lbl" not in variables:raise ValueError("Harvard缺ref/lbl")
        self.shape=variables["ref"][0]
        if len(self.shape)!=3 or self.shape[2]!=31 or variables["lbl"][0]!=self.shape[:2]:raise ValueError("Harvard实际维度不符")
        if self.calibration.shape!=(31,) or not np.isfinite(self.calibration).all() or min(self.calibration)<=0:raise ValueError("Harvard校正必须31个有限正值")
        if self.shape[0]*self.shape[1]*8>64*1024**2:raise ValueError("Harvard lbl超过当前有界读取器64MiB门槛，须扩展分块实现")
        labels=loadmat(path,variable_names=["lbl"],appendmat=False)["lbl"]
        self.mask=(labels!=0)&np.isfinite(labels)
        self.indices=wavelength_indices(np.arange(420,721,10));self.peak_block_bytes=0
        self.metadata={"shape_hwc":list(self.shape),"wavelengths_nm":list(range(420,721,10)),"selected_indices":self.indices,
                       "units":"相对能量谱密度，ref/相机相对灵敏度；绝对因子未知","calibration":"divide_sensitivity_no_photon_conversion","mask":"lbl非零且有限；所选全部谱带有限非负","reader":"MAT-v5流式64列块"}
    def blocks(self):
        for band,x,raw in matlab_ref_blocks(self.path,self.indices):
            self.peak_block_bytes=max(self.peak_block_bytes,raw.nbytes+self.mask.nbytes)
            good=self.mask[:,x:x+raw.shape[1]]&np.isfinite(raw)&(raw>=0)
            energy=np.asarray(raw,dtype=np.float64)/self.calibration[band]
            yield self.indices.index(band),0,x,energy,good
    def close(self):pass


class ICVL:
    def __init__(self,path):
        import h5py
        self.file=h5py.File(path,"r")
        if "rad" not in self.file or "bands" not in self.file:raise ValueError("ICVL缺实际rad/bands元数据")
        self.rad=self.file["rad"];wave=np.asarray(self.file["bands"]).reshape(-1)
        self.indices=wavelength_indices(wave)
        if self.rad.ndim!=3:raise ValueError("ICVL rad应三维")
        axes=[i for i,n in enumerate(self.rad.shape) if n==len(wave)]
        if len(axes)!=1 or axes[0] not in [0,2]:raise ValueError("ICVL谱轴不明确")
        self.axis=axes[0]
        self.h,self.w=(self.rad.shape[2],self.rad.shape[1]) if self.axis==0 else self.rad.shape[:2]
        self.shape=(self.h,self.w,len(wave));self.peak_block_bytes=0
        self.metadata={"shape_hwc":list(self.shape),"wavelengths_nm":wave.tolist(),"selected_indices":self.indices,
                       "units":"rad发布相对辐亮度；共同尺度是跨数据集实施假设，不是绝对标定",
                       "axis_rule":"MATLAB-v7.3轴逆序" if self.axis==0 else "实际HWC谱末轴","mask":"所选全部谱带有限非负","reader":"HDF5实际bands映射/64列空间块联合读取谱通道，输出块不超过64MiB"}
    def blocks(self):
        columns=min(64,self.w)
        spatial_axis=2 if self.axis==0 else 0
        preferred=self.rad.chunks[spatial_axis] if self.rad.chunks else 128
        max_rows=64*1024**2//(self.shape[2]*columns*self.rad.dtype.itemsize)
        rows=min(512,max(2,preferred),max_rows)//2*2
        if rows<2:raise ValueError("ICVL最小空间块超过有界读取预算")
        for y in range(0,self.h,rows):
            for x in range(0,self.w,columns):
                # 读取空间小块中的全部原谱通道，避免全谱压缩块按目标波段重复解压。
                block=np.asarray(self.rad[:,x:x+columns,y:y+rows]) if self.axis==0 else np.asarray(self.rad[y:y+rows,x:x+columns,:])
                self.peak_block_bytes=max(self.peak_block_bytes,block.nbytes)
                for j,band in enumerate(self.indices):
                    raw=block[band].T if self.axis==0 else block[:,:,band]
                    yield j,y,x,raw,np.isfinite(raw)&(raw>=0)
    def close(self):self.file.close()


class KAIST:
    def __init__(self,path):
        import OpenEXR,Imath
        self.file=OpenEXR.InputFile(str(path));header=self.file.header()
        self.names=exr_wavelength_names(header["channels"])
        box=header["dataWindow"];self.h=box.max.y-box.min.y+1;self.w=box.max.x-box.min.x+1;self.y_origin=box.min.y
        self.shape=(self.h,self.w,len(header["channels"]));self.pixel_type=Imath.PixelType(Imath.PixelType.FLOAT)
        for name in self.names:
            if header["channels"][name].xSampling!=1 or header["channels"][name].ySampling!=1:raise ValueError("EXR谱通道存在未审核子采样")
        self.peak_block_bytes=0
        self.metadata={"shape_hwc":[self.h,self.w,25],"channel_names":self.names,"selected_indices":"按w{波长}nm名称，不按通道位置",
                       "units":"相对Spectralon参考白的反射率，保留原值，不拟合场景尺度","mask":"所选全部谱通道有限非负",
                       "reader":"EXR旧扫描行API按全部25通道有界扫描行块，总输出块不超过64MiB","data_window_origin_y":self.y_origin}
    def blocks(self):
        rows=min(128,(64*1024**2//(25*4*self.w))//2*2)
        if rows<2:raise ValueError("EXR最小两行谱块超过64MiB，需另行审核分列解码")
        for y in range(0,self.h,rows):
            end=min(y+rows,self.h)
            # 同一压缩块只解码一次；通道名称顺序已在初始化时核对。
            parts=self.file.channels(self.names,self.pixel_type,self.y_origin+y,self.y_origin+end-1)
            if len(parts)!=25:raise ValueError("EXR返回谱通道数量不符")
            self.peak_block_bytes=max(self.peak_block_bytes,sum(map(len,parts)))
            for j,part in enumerate(parts):
                raw=np.frombuffer(part,dtype=np.float32).reshape(end-y,self.w)
                yield j,y,0,raw,np.isfinite(raw)&(raw>=0)
    def close(self):self.file.close()


def inspect_scene(reader):
    """只产生整图固定24×24重复元数据，不计算预测、成绩或选取ROI。"""
    h,w=reader.shape[:2];yy=np.linspace(0,h-1,24).astype(int);xx=np.linspace(0,w-1,24).astype(int)
    descriptor=np.zeros((25,24,24),dtype=np.float64);seen=np.zeros_like(descriptor,dtype=bool)
    for band,y,x,raw,good in reader.blocks():
        iy=np.flatnonzero((yy>=y)&(yy<y+raw.shape[0]));ix=np.flatnonzero((xx>=x)&(xx<x+raw.shape[1]))
        if len(iy) and len(ix):
            sample=raw[np.ix_(yy[iy]-y,xx[ix]-x)];valid=good[np.ix_(yy[iy]-y,xx[ix]-x)]
            descriptor[band][np.ix_(iy,ix)]=np.where(valid,sample,0)
            seen[band][np.ix_(iy,ix)]=True
    if not seen.all():raise ValueError("重复元数据的实际通道读取不完整")
    flat=descriptor.ravel();norm=float(np.linalg.norm(flat))
    if not np.isfinite(norm) or norm<=0:raise ValueError("重复核对样本无有限正能量")
    return (flat/norm).astype(np.float32),reader.metadata


def preprocess(reader,data_path,mask_path):
    """分组后精确2×2面积平均，所有组成像元均有效才保留有效掩码。"""
    h,w=reader.shape[:2];oh,ow=h//2,w//2
    output=np.lib.format.open_memmap(data_path,mode="w+",dtype=np.float32,shape=(25,oh,ow))
    mask=np.lib.format.open_memmap(mask_path,mode="w+",dtype=bool,shape=(oh,ow));mask[:]=True
    for band,y,x,raw,good in reader.blocks():
        height=min(raw.shape[0],2*oh-y);width=min(raw.shape[1],2*ow-x)
        if height<=0 or width<=0:continue
        if y%2 or x%2 or height%2 or width%2:raise ValueError("分块不能破坏2×2物理面积边界")
        values=np.where(good[:height,:width],raw[:height,:width],0).astype(np.float64,copy=False)
        half=values.reshape(height//2,2,width//2,2).mean((1,3))
        valid=good[:height,:width].reshape(height//2,2,width//2,2).all((1,3))
        output[band,y//2:y//2+height//2,x//2:x//2+width//2]=half
        mask[y//2:y//2+height//2,x//2:x//2+width//2]&=valid
    for y in range(0,oh,64):output[:,y:y+64]*=mask[y:y+64][None]
    output.flush();mask.flush()
    return {"shape_chw":[25,oh,ow],"dropped_odd_edge_yx":[h%2,w%2],"valid_pixels":int(mask.sum()),
            "area_rule":"每2×2原像元等面积平均，几何像元中心对齐；奇数末行/列丢弃并登记","peak_source_block_bytes":reader.peak_block_bytes}
