import openslide

path = "./data/LBM_Path/Path/140195.mrxs"
slide = openslide.open_slide(path)
print(slide.level_count)
print(slide.level_dimensions)
