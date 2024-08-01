import openslide

path = "./data/SVS/Path/130445 1.svs"
slide = openslide.open_slide(path)
print(slide.level_count)
print(slide.level_dimensions)
