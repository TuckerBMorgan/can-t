use ggus::{GGmlType, GGuf};
use memmap2::Mmap;
use serde_json::Value;
use std::{fs::File, path::Path, slice::from_raw_parts};

pub struct GGUFFile {
    gguf: GGuf<'static>,
    _mmap: Mmap,
}

impl GGUFFile {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let file = File::open(path).unwrap();
        let file = unsafe { Mmap::map(&file) }.unwrap();
        let gguf = GGuf::new(unsafe { from_raw_parts(file.as_ptr(), file.len()) }).unwrap();
        Self { gguf, _mmap: file }
    }

    pub fn get_value(&self, name: impl AsRef<str>) -> Value {
        let meta_val = &self.gguf.meta_kvs[name.as_ref()];
        let mut reader = meta_val.value_reader();
        use ggus::GGufMetaDataValueType as Ty;
        match meta_val.ty() {
            Ty::U8 => reader.read::<u8>().unwrap().into(),
            Ty::I8 => reader.read::<i8>().unwrap().into(),
            Ty::U16 => reader.read::<u16>().unwrap().into(),
            Ty::I16 => reader.read::<i16>().unwrap().into(),
            Ty::U32 => reader.read::<u32>().unwrap().into(),
            Ty::I32 => reader.read::<i32>().unwrap().into(),
            Ty::F32 => reader.read::<f32>().unwrap().into(),
            Ty::Bool => reader.read::<bool>().unwrap().into(),
            Ty::String => reader.read_str().unwrap().into(),
            Ty::U64 => reader.read::<u64>().unwrap().into(),
            Ty::I64 => reader.read::<i64>().unwrap().into(),
            Ty::F64 => reader.read::<f64>().unwrap().into(),
            Ty::Array => todo!("array meta data"),
        }
    }

    pub fn get_tensor_dims(&self, name: impl AsRef<str>) -> Vec<usize> {
        self.gguf.tensors[name.as_ref()]
            .to_info()
            .shape()
            .iter()
            .rev()
            .map(|&d| d as _)
            .collect()
    }

    pub fn get_weight_for_tensor(&self, name: impl AsRef<str>) -> Vec<f32> {
        let tensor = self.gguf.tensors[name.as_ref()].to_info();
        match tensor.ty() {
            GGmlType::F16 => {
                let len = tensor.shape().iter().product::<u64>() as usize;
                let data = &self.gguf.data[tensor.offset()..][..len * size_of::<f16>()];
                let ([], data, []) = (unsafe { data.align_to::<f16>() }) else {
                    unreachable!()
                };
                return data
                    .iter()
                    .map(|x| *x as f32)
                    .collect::<Vec<f32>>()
                    .to_vec();
            }
            GGmlType::F32 => {
                let len = tensor.shape().iter().product::<u64>() as usize;
                let data = &self.gguf.data[tensor.offset()..][..len * size_of::<f32>()];
                let ([], data, []) = (unsafe { data.align_to::<f32>() }) else {
                    unreachable!()
                };
                return data.to_vec();
            }
            _ => {
                panic!("{:?}: unsupported data data", tensor.ty());
            }
        }
    }
}
