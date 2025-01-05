from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import local

# this gets all the connected regions and groups them together
label_image = measure.label(local.binimg)
fig, (ax1) = plt.subplots(1)
ax1.imshow(local.grayimg, cmap="gray")

# regionprops creates a list of properties of all the labelled regions
for region in regionprops(label_image):
    if region.area < 1000:
        #if the region is so small then it's likely not a license plate
        continue

    # the bounding box coordinates
    minRow, minCol, maxRow, maxCol = region.bbox
    rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)
    ax1.add_patch(rectBorder)
    # let's draw a red rectangle over those regions

plt.show()