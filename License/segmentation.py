import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import ccaimpr
import matplotlib.pyplot as plt
import matplotlib.patches as patches


licenseplate=np.invert(ccaimpr.platelikeobs[1])
labelplate=measure.label(licenseplate)
fig,ax1=plt.subplots()
ax1.imshow(licenseplate,cmap="gray")
chardimensions=(0.35*licenseplate.shape[0],0.60*licenseplate.shape[0],0.05*licenseplate.shape[1],0.15*licenseplate.shape[1])
minheight,maxheight,minwidth,maxwidth=chardimensions
character=[]
counter=0
columnlist=[]
for region in regionprops(labelplate):
    y0,x0,y1,x1=region.bbox
    regionh=y1-y0
    regionw=x1-x0
    if minheight<regionh<maxheight and minwidth<regionw<maxwidth:
        roi=licenseplate[y0:y1,x0:x1]
        rectBorder=patches.Rectangle((x0,y0),x1-x0,y1-y0,edgecolor='red',linewidth=2,fill=False)
        ax1.add_patch(rectBorder)
        resizedchar=resize(roi,(20,20))
        character.append(resizedchar)
        columnlist.append(x0)
plt.show()