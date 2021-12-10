from nasaimages.openimage import OpenImage
from nasaimages.wisephoto import WisePhoto

# use path for specific image
PATH = "./out/2021-12-09.png"

# use APOD for today's NASA-provided image
apod = WisePhoto()

# OpenImage can be instatitated with a path (str) or image matrix
# oi = OpenImage(PATH)
oi = OpenImage(apod.photo)
oi.show('img', oi.img)