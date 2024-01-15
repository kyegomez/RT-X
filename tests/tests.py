import pytest
import torch
from PIL import Image
from zeta.structs import (
    AutoregressiveWrapper,
    ViTransformerWrapper,
)

from rtx.efficient_net import EfficientNetFilm
from rtx.rtx1 import RT1, RTX1, FilmMaxVit
from rtx.rtx2 import RTX2


########################### EfficientNetFilm ###########################
img = "img.jpeg"


# Fixture to create an instance of the EfficientNetFilm class
@pytest.fixture
def efficientnet_model():
    model = EfficientNetFilm("efficientnet-b0", 10)
    return model


# Test case to check if EfficientNetFilm initializes correctly
def test_efficientnet_init(efficientnet_model):
    assert efficientnet_model is not None


# Test case to check if EfficientNetFilm processes an image correctly
def test_efficientnet_process_image(efficientnet_model):
    # Load a sample image
    image_path = img
    Image.open(image_path)

    # Process the image using the model
    features = efficientnet_model(image_path)

    # Check if the output features are of the correct shape
    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, efficientnet_model.num_features)


# Test case to check if EfficientNetFilm handles image resizing correctly
def test_efficientnet_image_resize(efficientnet_model):
    # Load a sample image
    image_path = img
    image = Image.open(image_path)

    # Process the image using the model
    efficientnet_model(image_path)

    # Check if the input image was resized to the specified size
    assert image.size == (
        efficientnet_model.resize,
        efficientnet_model.resize,
    )


# Test case to check if EfficientNetFilm handles model loading correctly
def test_efficientnet_model_loading(efficientnet_model):
    # Check if the model was loaded successfully
    assert efficientnet_model.model is not None


# Test case to check if EfficientNetFilm handles image transformations correctly
def test_efficientnet_image_transformations(efficientnet_model):
    # Load a sample image
    image_path = img
    Image.open(image_path)

    # Process the image using the model
    features = efficientnet_model(image_path)

    # Check if image transformations were applied correctly
    assert torch.max(features).item() <= 1.0
    assert torch.min(features).item() >= -1.0


# Test case to check if EfficientNetFilm handles the number of classes correctly
def test_efficientnet_num_classes(efficientnet_model):
    # Check if the number of classes is set correctly
    assert efficientnet_model.num_classes == 10


# Test case to check if EfficientNetFilm handles missing image file correctly
def test_efficientnet_missing_image(efficientnet_model):
    with pytest.raises(FileNotFoundError):
        efficientnet_model("non_existent_image.jpg")


# Test case to check if EfficientNetFilm handles incorrect image file format correctly
def test_efficientnet_incorrect_image_format(efficientnet_model):
    with pytest.raises(ValueError):
        efficientnet_model("sample_image.txt")


# Test case to check if EfficientNetFilm handles model selection correctly
def test_efficientnet_model_selection():
    # Check if different EfficientNet models can be selected
    model_names = [
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
    ]
    for model_name in model_names:
        model = EfficientNetFilm(model_name, 10)
        assert model is not None
        assert model.model is not None


# Test case to check if EfficientNetFilm handles invalid model name correctly
def test_efficientnet_invalid_model_name():
    with pytest.raises(ValueError):
        EfficientNetFilm("invalid_model", 10)


# Test case to check if EfficientNetFilm handles invalid number of classes correctly
def test_efficientnet_invalid_num_classes():
    with pytest.raises(ValueError):
        EfficientNetFilm("efficientnet-b0", -10)


# Test case to check if EfficientNetFilm handles invalid resize size correctly
def test_efficientnet_invalid_resize_size():
    with pytest.raises(ValueError):
        EfficientNetFilm("efficientnet-b0", 10, resize=-100)


# Test case to check if EfficientNetFilm handles input image with incorrect channels correctly
def test_efficientnet_incorrect_image_channels(efficientnet_model):
    # Create an image with incorrect number of channels (4 channels)
    image = Image.new(
        "RGBA",
        (efficientnet_model.resize, efficientnet_model.resize),
        (255, 0, 0, 255),
    )
    image_path = "incorrect_channels_image.png"
    image.save(image_path)

    with pytest.raises(ValueError):
        efficientnet_model(image_path)


# Test case to check if EfficientNetFilm handles input image with incorrect size correctly
def test_efficientnet_incorrect_image_size(efficientnet_model):
    # Create an image with incorrect size (smaller than resize size)
    image = Image.new(
        "RGB",
        (
            efficientnet_model.resize - 1,
            efficientnet_model.resize - 1,
        ),
        (255, 0, 0),
    )
    image_path = "incorrect_size_image.jpg"
    image.save(image_path)

    with pytest.raises(ValueError):
        efficientnet_model(image_path)


########################### RTX1 ###########################


# Fixture to create an instance of the RTX1 class
@pytest.fixture
def rtx1_model():
    model = RTX1()
    return model


# Test case to check if RTX1 initializes correctly
def test_rtx1_initialization(rtx1_model):
    assert isinstance(rtx1_model, RTX1)
    assert isinstance(rtx1_model.vit, FilmMaxVit)
    assert isinstance(rtx1_model.model, RT1)


# Test case to check if RTX1 handles training with video and instructions correctly
def test_rtx1_train(rtx1_model):
    video = torch.randn(2, 3, 6, 224, 224)
    instructions = [
        "bring me that apple sitting on the table",
        "please pass the butter",
    ]

    train_logits = rtx1_model.train(video, instructions)

    assert isinstance(train_logits, torch.Tensor)
    assert train_logits.shape == (2, rtx1_model.num_actions)


# Test case to check if RTX1 handles evaluation with video and instructions correctly
def test_rtx1_eval(rtx1_model):
    video = torch.randn(2, 3, 6, 224, 224)
    instructions = [
        "bring me that apple sitting on the table",
        "please pass the butter",
    ]

    eval_logits = rtx1_model.run(video, instructions, cond_scale=3.0)

    assert isinstance(eval_logits, torch.Tensor)
    assert eval_logits.shape == (2, rtx1_model.num_actions)


# Test case to check if RTX1 raises an error when training with invalid inputs
def test_rtx1_train_with_invalid_inputs(rtx1_model):
    with pytest.raises(RuntimeError):
        video = torch.randn(2, 3, 6, 224, 224)
        instructions = [
            "bring me that apple sitting on the table",
            "please pass the butter",
        ]
        # Intentionally set an invalid shape for instructions
        instructions = instructions[:1]  # Instructions shape should be (2,)
        rtx1_model.train(video, instructions)


# Test case to check if RTX1 raises an error when evaluating with invalid inputs
def test_rtx1_eval_with_invalid_inputs(rtx1_model):
    with pytest.raises(RuntimeError):
        video = torch.randn(2, 3, 6, 224, 224)
        instructions = [
            "bring me that apple sitting on the table",
            "please pass the butter",
        ]
        # Intentionally set an invalid shape for video
        video = video[:, :, :5]  # Video shape should be (2, 3, 6, 224, 224)
        rtx1_model.run(video, instructions, cond_scale=3.0)


# Test case to check if RTX1 handles conditional scaling correctly
def test_rtx1_conditional_scaling(rtx1_model):
    video = torch.randn(2, 3, 6, 224, 224)
    instructions = [
        "bring me that apple sitting on the table",
        "please pass the butter",
    ]

    eval_logits = rtx1_model.run(video, instructions, cond_scale=3.0)
    eval_logits_without_scaling = rtx1_model.run(video, instructions)

    # Check if the logits with and without scaling are different
    assert not torch.allclose(eval_logits, eval_logits_without_scaling)


# Test case to check if RTX1 handles model selection correctly
def test_rtx1_model_selection():
    model_names = [
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
    ]
    for model_name in model_names:
        model = RTX1(model_name=model_name)
        assert isinstance(model, RTX1)


# Test case to check if RTX1 raises an error for an invalid model name
def test_rtx1_invalid_model_name():
    with pytest.raises(ValueError):
        RTX1(model_name="invalid_model")


# Test case to check if RTX1 handles negative number of classes correctly
def test_rtx1_negative_num_classes():
    with pytest.raises(ValueError):
        RTX1(num_classes=-100)


# Test case to check if RTX1 handles negative dimension correctly
def test_rtx1_negative_dimension():
    with pytest.raises(ValueError):
        RTX1(dim=-96)


# Test case to check if RTX1 handles negative dimension of convolutional stem correctly
def test_rtx1_negative_dim_conv_stem():
    with pytest.raises(ValueError):
        RTX1(dim_conv_stem=-64)


# Test case to check if RTX1 handles negative dimension of head for ViT correctly
def test_rtx1_negative_dim_head_vit():
    with pytest.raises(ValueError):
        RTX1(dim_head_vit=-32)


# Test case to check if RTX1 handles negative depth of ViT correctly
def test_rtx1_negative_depth_vit():
    with pytest.raises(ValueError):
        RTX1(depth_vit=(-2, 2, 5, 2))


# Test case to check if RTX1 handles negative window size for ViT correctly
def test_rtx1_negative_window_size():
    with pytest.raises(ValueError):
        RTX1(window_size=-7)


# Test case to check if RTX1 handles negative expansion rate for mbconv correctly
def test_rtx1_negative_mbconv_expansion_rate():
    with pytest.raises(ValueError):
        RTX1(mbconv_expansion_rate=-4)


# Test case to check if RTX1 handles negative shrinkage rate for mbconv correctly
def test_rtx1_negative_mbconv_shrinkage_rate():
    with pytest.raises(ValueError):
        RTX1(mbconv_shrinkage_rate=-0.25)


# Test case to check if RTX1 handles negative dropout rate for ViT correctly
def test_rtx1_negative_dropout_vit():
    with pytest.raises(ValueError):
        RTX1(dropout_vit=-0.1)


# Test case to check if RTX1 handles negative number of actions correctly
def test_rtx1_negative_num_actions():
    with pytest.raises(ValueError):
        RTX1(num_actions=-11)


# Test case to check if RTX1 handles negative depth of RT1 correctly
def test_rtx1_negative_depth_rt1():
    with pytest.raises(ValueError):
        RTX1(depth_rt1=-6)


# Test case to check if RTX1 handles negative number of heads for RT1 correctly
def test_rtx1_negative_heads():
    with pytest.raises(ValueError):
        RTX1(heads=-8)


# Test case to check if RTX1 handles negative dimension of head for RT1 correctly
def test_rtx1_negative_dim_head_rt1():
    with pytest.raises(ValueError):
        RTX1(dim_head_rt1=-64)


# Test case to check if RTX1 handles negative conditional drop probability for RT1 correctly
def test_rtx1_negative_cond_drop_prob():
    with pytest.raises(ValueError):
        RTX1(cond_drop_prob=-0.2)


########################### RTX2 ###########################


# Fixture to create an instance of the RTX2 class
@pytest.fixture
def rtx2_model():
    model = RTX2()
    return model


# Test case to check if RTX2 initializes correctly
def test_rtx2_initialization(rtx2_model):
    assert isinstance(rtx2_model, RTX2)
    assert isinstance(rtx2_model.encoder, ViTransformerWrapper)
    assert isinstance(rtx2_model.decoder, AutoregressiveWrapper)


# Test case to check if RTX2 handles forward pass with image and text correctly
def test_rtx2_forward_pass(rtx2_model):
    img = torch.randn(1, 3, 256, 256)
    text = torch.randint(0, 20000, (1, 1024))

    output = rtx2_model(img, text)

    assert isinstance(output, torch.Tensor)


# Test case to check if RTX2 raises an error when forwarding with invalid inputs
def test_rtx2_forward_with_invalid_inputs(rtx2_model):
    with pytest.raises(Exception):
        img = torch.randn(1, 3, 256, 256)
        text = torch.randn(1, 1024, 512)  # Invalid shape for text input
        rtx2_model(img, text)


# Test case to check if RTX2 handles various model configurations correctly
def test_rtx2_with_different_configs():
    config_combinations = [
        {"encoder_depth": 6, "decoder_depth": 6},
        {"encoder_depth": 4, "decoder_depth": 8},
        {"encoder_heads": 8, "decoder_heads": 8},
        {"encoder_dim": 512, "decoder_dim": 768},
    ]

    for config in config_combinations:
        model = RTX2(**config)
        assert isinstance(model, RTX2)
        assert model.encoder.attn_layers.depth == config["encoder_depth"]
        assert model.decoder.attn_layers.depth == config["decoder_depth"]
        if "encoder_heads" in config:
            assert model.encoder.attn_layers.heads == config["encoder_heads"]
        if "decoder_heads" in config:
            assert model.decoder.attn_layers.heads == config["decoder_heads"]
        if "encoder_dim" in config:
            assert model.encoder.attn_layers.dim == config["encoder_dim"]
        if "decoder_dim" in config:
            assert model.decoder.attn_layers.dim == config["decoder_dim"]


# Test case to check if RTX2 handles negative image size correctly
def test_rtx2_negative_image_size():
    with pytest.raises(ValueError):
        RTX2(image_size=-256)


# Test case to check if RTX2 handles negative patch size correctly
def test_rtx2_negative_patch_size():
    with pytest.raises(ValueError):
        RTX2(patch_size=-32)


# Test case to check if RTX2 handles negative encoder dimension correctly
def test_rtx2_negative_encoder_dim():
    with pytest.raises(ValueError):
        RTX2(encoder_dim=-512)


# Test case to check if RTX2 handles negative encoder depth correctly
def test_rtx2_negative_encoder_depth():
    with pytest.raises(ValueError):
        RTX2(encoder_depth=-6)


# Test case to check if RTX2 handles negative decoder dimension correctly
def test_rtx2_negative_decoder_dim():
    with pytest.raises(ValueError):
        RTX2(decoder_dim=-512)


# Test case to check if RTX2 handles negative decoder depth correctly
def test_rtx2_negative_decoder_depth():
    with pytest.raises(ValueError):
        RTX2(decoder_depth=-6)


# Test case to check if RTX2 handles negative encoder heads correctly
def test_rtx2_negative_encoder_heads():
    with pytest.raises(ValueError):
        RTX2(encoder_heads=-8)


# Test case to check if RTX2 handles negative decoder heads correctly
def test_rtx2_negative_decoder_heads():
    with pytest.raises(ValueError):
        RTX2(decoder_heads=-8)
