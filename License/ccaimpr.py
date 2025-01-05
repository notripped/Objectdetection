from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import local

labelimg=measure.label(local.binimg)
platedimensions = (0.08*labelimg.shape[0], 0.2*labelimg.shape[0], 0.15*labelimg.shape[1], 0.4*labelimg.shape[1])
minheight,maxheight,minwidth,maxwidth=platedimensions
plateobjectcoords=[]
platelikeobs=[]
fig,(ax1)=plt.subplots(1)
ax1.imshow(local.grayimg,cmap="gray")
for region in regionprops(labelimg):
    if region.area<50:
        continue
    minrow,mincol,maxrow,maxcol=region.bbox
    regionh=maxrow-minrow
    regionw=maxcol-mincol
    if minheight <= regionh <= maxheight and minwidth <= regionw <= maxwidth and regionw>regionh:
        platelikeobs.append(local.binimg[minrow:maxrow,mincol:maxcol])
        plateobjectcoords.append((minrow,mincol,maxrow,maxcol))
        rectBorder=patches.Rectangle((mincol,minrow),maxcol-mincol,maxrow-minrow,edgecolor="red",linewidth=2,fill=False)
        ax1.add_patch(rectBorder)
print(platelikeobs)
plt.show()