use anyhow::{Result, anyhow};
use serde_json::Value;
use std::collections::HashMap;
use std::io::prelude::*;
use std::fs::File;

    #[derive(Debug)]
pub enum MetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for MetadataValueType {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => MetadataValueType::Uint8,
            1 => MetadataValueType::Int8,
            2 => MetadataValueType::Uint16,
            3 => MetadataValueType::Int16,
            4 => MetadataValueType::Uint32,
            5 => MetadataValueType::Int32,
            6 => MetadataValueType::Float32,
            7 => MetadataValueType::Bool,
            8 => MetadataValueType::String,
            9 => MetadataValueType::Array,
            10 => MetadataValueType::Uint64,
            11 => MetadataValueType::Int64,
            12 => MetadataValueType::Float64,
            _ => return Err(anyhow!("unsupport metadata value type")),
        })
    }
}

fn read_u8(f: &mut File) -> u8 {
    let mut u8_buffer = [0;1];
    f.read_exact(&mut u8_buffer);
    return u8::from_le(u8_buffer[0]);
}

fn read_i8(f: &mut File) -> i8 {
    let mut u8_buffer = [0;1];
    f.read_exact(&mut u8_buffer);
    return i8::from_le_bytes(u8_buffer);
}

fn read_u16(f: &mut File) -> u16 {

    let mut u8_buffer = [0;2];
    f.read_exact(&mut u8_buffer);
    return u16::from_le_bytes(u8_buffer);
}

fn read_i16(f: &mut File) -> i16 {

    let mut u8_buffer = [0;2];
    f.read_exact(&mut u8_buffer);
    return i16::from_le_bytes(u8_buffer);
}

fn read_u32(f: &mut File) -> u32 {

    let mut u8_buffer = [0;4];
    f.read_exact(&mut u8_buffer);
    return u32::from_le_bytes(u8_buffer);
}

fn read_u64(f: &mut File) -> u64 {

    let mut u8_buffer = [0;8];
    f.read_exact(&mut u8_buffer);
    return u64::from_le_bytes(u8_buffer);
}

fn read_i32(f: &mut File) -> i32 {

    let mut u8_buffer = [0;4];
    f.read_exact(&mut u8_buffer);
    return i32::from_le_bytes(u8_buffer);
}

fn read_f32(f: &mut File) -> f32 {

    let mut u8_buffer = [0;4];
    f.read_exact(&mut u8_buffer);
    return f32::from_le_bytes(u8_buffer);
}

fn read_bool(f: &mut File) -> bool {

    let mut u8_buffer = [0;1];
    f.read_exact(&mut u8_buffer);
    return u8::from_le(u8_buffer[0]) != 0;
}

fn read_string(f: &mut File) -> String {

    let mut string_size = [0;8];
    f.read_exact(&mut string_size);
    let string_size = u64::from_le_bytes(string_size);
    let mut string_buffer = vec![0;string_size as usize];
    f.read_exact(&mut string_buffer);
    return String::from_utf8_lossy(&string_buffer).to_string();
}

fn read_array(f: &mut File, read_count: usize) -> Vec<Value> {
    let mut data = vec![];
    let item_type = read_u32(f).try_into().unwrap();
    let array_len = read_u64(f);
    for _ in 0.. array_len {
           let value =  match item_type {
                MetadataValueType::Uint8 =>{
                     Value::from(read_u8(f))
                },
                MetadataValueType::Int8 =>{ 
                    Value::from(read_i8(f))
                },
                MetadataValueType::Uint16 => Value::from(read_u16(f)),
                MetadataValueType::Int16 => Value::from(read_i16(f)),
                MetadataValueType::Uint32 => Value::from(read_u32(f)),
                MetadataValueType::Int32 => Value::from(read_i32(f)),
                MetadataValueType::Float32 => Value::from(read_f32(f)),
                MetadataValueType::Bool => Value::from(read_bool(f)),
                MetadataValueType::String => Value::from(read_string(f)),
                MetadataValueType::Array => {panic!("Cant embded arrays in array");/*read_array(&mut f, 3);*/},
                MetadataValueType::Uint64 => {panic!("u64");/*read_u64(&mut f);*/},
                MetadataValueType::Int64 => {panic!("i64");/*read_i64(&mut f);*/},
                MetadataValueType::Float64 => {panic!("f64");/*read_f64(&mut f);*/},
            };   
        if read_count > 0 && data.len() < read_count {
            data.push(value);
        }
    }


    return data;
}

fn align_offset(offset: u64) -> u64{
    return offset + (32 - (offset % 32)) % 32;
}

pub struct GGUFFile {
    tensors: Vec<TensorData>,
    key_value_pairs: HashMap<String, Value>,
    data_start: u64
}

struct TensorData {
    name: String,
    dimensions: Vec<u64>,
    offset: u64
}

impl TensorData {
    pub fn new(name: String, dimensions: Vec<u64>, offset: u64) -> TensorData {
        TensorData {
            name,
            dimensions,
            offset
        }
    }
}
impl GGUFFile {
 fn new(path: String) -> GGUFFile {
        let mut f = File::open(path.as_str()).unwrap();
        let mut magic_number_buffer = [0;4];
        let _magic_number_error = f.read_exact(&mut magic_number_buffer);

        let mut version = [0;4];
        let _version_number_error = f.read_exact(&mut version);
        let _as_number = u32::from_le_bytes([version[0], version[1], version[2], version[3]]);

        let mut number_of_tensors =[0;8];
        let _number_of_tensors_error = f.read_exact(&mut number_of_tensors);
        let array_of_bytes = [number_of_tensors[0], number_of_tensors[1], number_of_tensors[2], number_of_tensors[3], number_of_tensors[4], number_of_tensors[5], number_of_tensors[6], number_of_tensors[7]];
        let number_of_tensor = u64::from_le_bytes(array_of_bytes);

        let mut number_of_keyvalue_pairs = [0;8];
        let _number_of_key_value_error = f.read_exact(&mut number_of_keyvalue_pairs);
        let array_of_bytes = [number_of_keyvalue_pairs[0], number_of_keyvalue_pairs[1], number_of_keyvalue_pairs[2], number_of_keyvalue_pairs[3], number_of_keyvalue_pairs[4], number_of_keyvalue_pairs[5], number_of_keyvalue_pairs[6], number_of_keyvalue_pairs[7]];
        let number_of_keyvalue_pairs = u64::from_le_bytes(array_of_bytes);

        let mut key_value_pairs = HashMap::new();
        for _i in 0..number_of_keyvalue_pairs {
            let key = read_string(&mut f);
            let mut value_type = [0;4];
            let _value_type_error = f.read_exact(&mut value_type);
            let value_type = u32::from_le_bytes([value_type[0], value_type[1], value_type[2], value_type[3]]).try_into().unwrap();

            let value =  match value_type {
                MetadataValueType::Uint8 =>{
                    Value::from(read_u8(&mut f))
                },
                MetadataValueType::Int8 =>{ 
                    Value::from(read_i8(&mut f))
                },
                MetadataValueType::Uint16 => {Value::from(read_u16(&mut f))},
                MetadataValueType::Int16 => {Value::from(read_i16(&mut f))},
                MetadataValueType::Uint32 => {Value::from(read_u32(&mut f))},
                MetadataValueType::Int32 => {Value::from(read_i32(&mut f))},
                MetadataValueType::Float32 => {Value::from(read_f32(&mut f))},
                MetadataValueType::Bool => {Value::from(read_bool(&mut f))},
                MetadataValueType::String => {Value::from(read_string(&mut f))},
                MetadataValueType::Array => {Value::from(read_array(&mut f, 3))},
                MetadataValueType::Uint64 => {panic!("u64");/*read_u64(&mut f);*/},
                MetadataValueType::Int64 => {panic!("i64");/*read_i64(&mut f);*/},
                MetadataValueType::Float64 => {panic!("f64");/*read_f64(&mut f);*/},
            };
            key_value_pairs.insert(key, value);
        }

        let mut tensors = vec![];
        for _i in 0..number_of_tensor {
            let tensor_name = read_string(&mut f);
            let n_dimension = read_u32(&mut f);
            let mut dimensions = vec![];
            for _n in 0..n_dimension {
                let dimension = read_u64(&mut f);
                dimensions.push(dimension);
            }
            let _tensor_type = read_u32(&mut f);
            let offset = read_u64(&mut f);
            let tensor = TensorData::new(tensor_name, dimensions, offset);
            tensors.push(tensor);
        }

        let position = f.stream_position().unwrap();
        let padding = align_offset(position) - position;
        println!("{:?}", padding);

        f.seek_relative(padding as i64);
        GGUFFile {
            tensors,
            key_value_pairs,
            data_start: f.stream_position().unwrap()
        }
    }
}