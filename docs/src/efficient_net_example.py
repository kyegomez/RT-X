from rtx.efficient_net import EfficientNetFilm

model = EfficientNetFilm("efficientnet-b0", 10)

out = model("img.jpeg")
