import numpy as np
from PIL import Image
import io
import torch
from rtx import data_util

def test_describe():
    dic = {'key1': 'value1', 'key2': [1, 2, 3], 'key3': {'nested_key': 'nested_value'}}
    data_util.describe(dic)

def test_describe_empty():
    dic = {}
    data_util.describe(dic)

def test_describe_non_dict():
    non_dict = 'not a dict'
    data_util.describe(non_dict)

def test_preprocess():
    dic = {'key1': 'value1', 'key2': [1, 2, 3], 'key3': {'nested_key': 'nested_value'}}
    result = data_util.preprocess(dic)
    assert result == dic

def test_preprocess_empty():
    dic = {}
    result = data_util.preprocess(dic)
    assert result == dic

def test_preprocess_non_dict():
    non_dict = 'not a dict'
    result = data_util.preprocess(non_dict)
    assert result == non_dict

def test_preprocess_none_value():
    dic = {'key1': None}
    result = data_util.preprocess(dic)
    assert result == {}

def test_preprocess_image():
    img = Image.new('RGB', (60, 30), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    result = data_util.preprocess(img_byte_arr)
    assert isinstance(result, np.ndarray)

def test_format_imgs():
    dic = {'key1': 'value1', 'key2': [1, 2, 3], 'key3': {'nested_key': 'nested_value'}}
    result = data_util.format_imgs(dic, 224)
    assert result == dic

def test_format_imgs_empty():
    dic = {}
    result = data_util.format_imgs(dic, 224)
    assert result == dic

def test_format_imgs_non_dict():
    non_dict = 'not a dict'
    result = data_util.format_imgs(non_dict, 224)
    assert result == non_dict

def test_format_imgs_image():
    img = Image.new('RGB', (60, 30), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    result = data_util.format_imgs(img_byte_arr, 224)
    assert isinstance(result, np.ndarray)

def test_format_imgs_tensor():
    tensor = torch.tensor([1, 2, 3])
    result = data_util.format_imgs(tensor, 224)
    assert isinstance(result, torch.Tensor)

def test_format_imgs_list():
    list_val = [1, 2, 3]
    result = data_util.format_imgs(list_val, 224)
    assert result == list_val

def test_format_imgs_nested_dict():
    dic = {'key1': {'nested_key': 'nested_value'}}
    result = data_util.format_imgs(dic, 224)
    assert result == dic

def test_format_imgs_nested_list():
    dic = {'key1': [1, 2, 3]}
    result = data_util.format_imgs(dic, 224)
    assert result == dic

def test_format_imgs_nested_image():
    img = Image.new('RGB', (60, 30), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    dic = {'key1': img_byte_arr}
    result = data_util.format_imgs(dic, 224)
    assert isinstance(result['key1'], np.ndarray)

def test_format_imgs_nested_tensor():
    tensor = torch.tensor([1, 2, 3])
    dic = {'key1': tensor}
    result = data_util.format_imgs(dic, 224)
    assert isinstance(result['key1'], torch.Tensor)

def test_format_imgs_nested_list():
    list_val = [1, 2, 3]
    dic = {'key1': list_val}
    result = data_util.format_imgs(dic, 224)
    assert result == dic