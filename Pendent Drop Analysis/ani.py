import os, shutil, math
import numpy as np
import cv2
import matplotlib.pyplot as plt

output_dir='intermed'
if os.path.exists(output_dir): shutil.rmtree(output_dir)
os.makedirs(output_dir,exist_ok=True)

image_path='/Users/yash/Desktop/Disa Images/DISA Code/Pendent Drop Analysis/photogoutte3.jpg'
input_bgr=cv2.imread(image_path)
gray_image=cv2.cvtColor(input_bgr,cv2.COLOR_BGR2GRAY)
blurred_image=cv2.GaussianBlur(gray_image,(5,5),0)
_,binary_inverted=cv2.threshold(blurred_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
closing_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
closed_mask=cv2.morphologyEx(binary_inverted,cv2.MORPH_CLOSE,closing_kernel,iterations=3)
contours,_=cv2.findContours(closed_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
largest_contour=max(contours,key=cv2.contourArea)
filled_mask=np.zeros_like(closed_mask); cv2.drawContours(filled_mask,[largest_contour],-1,255,-1)
boundary_mask=cv2.morphologyEx(filled_mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))

cv2.imwrite(os.path.join(output_dir,'01_original.png'),input_bgr)
cv2.imwrite(os.path.join(output_dir,'02_gray.png'),gray_image)
cv2.imwrite(os.path.join(output_dir,'03_blur.png'),blurred_image)
cv2.imwrite(os.path.join(output_dir,'04_otsu_inv.png'),binary_inverted)
cv2.imwrite(os.path.join(output_dir,'05_closed.png'),closed_mask)
cv2.imwrite(os.path.join(output_dir,'06_filled_mask.png'),filled_mask)
cv2.imwrite(os.path.join(output_dir,'07_boundary.png'),boundary_mask)

x0,y0,w0,h0=cv2.boundingRect(largest_contour)
roi_top=y0
roi_bottom=y0+int(h0*0.8)
row_y=np.arange(roi_top,roi_bottom)
row_left=np.full_like(row_y,np.nan,dtype=float)
row_right=np.full_like(row_y,np.nan,dtype=float)
for i,yy in enumerate(row_y):
    xs=np.where(boundary_mask[yy,x0:x0+w0]>0)[0]
    if xs.size:
        row_left[i]=x0+xs.min()
        row_right[i]=x0+xs.max()
row_width=row_right-row_left
valid_idx=np.where(np.isfinite(row_width))[0]
k=min(9,max(3,(len(valid_idx)//5)*2+1))
pad_k=k//2
pad=np.pad(row_width[valid_idx],(pad_k,pad_k),mode='edge')
med=np.array([np.median(pad[i:i+k]) for i in range(pad.size-k+1)])
row_width_med=np.full_like(row_width,np.nan)
row_width_med[valid_idx]=med
head_n=min(120,len(valid_idx)//3+20)
head_idx=valid_idx[:head_n]
base_width=np.median(row_width_med[head_idx])
tol=max(3.0,0.08*base_width)
consecutive=12
break_pos=valid_idx[-1]
run=0
for idx in valid_idx:
    if row_width_med[idx]>base_width+tol:
        run+=1
        if run>=consecutive:
            break_pos=idx-consecutive//2
            break
    else:
        run=0
plate_idx=valid_idx[valid_idx<break_pos]
plate_y=row_y[plate_idx]
plate_left=row_left[plate_idx]
plate_right=row_right[plate_idx]

plt.figure(figsize=(5,6))
plt.plot(row_width,row_y,label='w')
plt.plot(row_width_med,row_y,label='w_med')
plt.axvline(base_width,ls='--')
plt.axvline(base_width+tol,ls='--')
plt.axhline(row_y[break_pos],ls='--')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'08_width_profile.png')); plt.close()

points_vis=input_bgr.copy()
for xx,yy in zip(plate_left.astype(int),plate_y.astype(int)): cv2.circle(points_vis,(xx,yy),2,(0,0,255),-1)
for xx,yy in zip(plate_right.astype(int),plate_y.astype(int)): cv2.circle(points_vis,(xx,yy),2,(255,0,0),-1)
cv2.imwrite(os.path.join(output_dir,'09_plateau_points.png'),points_vis)

coef_left=np.polyfit(plate_y,plate_left,1)
coef_right=np.polyfit(plate_y,plate_right,1)
aL,bL=coef_left; aR,bR=coef_right
def x_at_y(a,b,y): return a*y+b
lines_vis=input_bgr.copy()
yy1=y0; yy2=y0+h0
cv2.line(lines_vis,(int(x_at_y(aL,bL,yy1)),yy1),(int(x_at_y(aL,bL,yy2)),yy2),(0,255,255),2)
cv2.line(lines_vis,(int(x_at_y(aR,bR,yy1)),yy1),(int(x_at_y(aR,bR,yy2)),yy2),(0,255,255),2)
cv2.imwrite(os.path.join(output_dir,'10_fitted_lines_plateau.png'),lines_vis)

def nearest_boundary_distance(q,normal_dir,search_radius=30):
    n=normal_dir/np.linalg.norm(normal_dir)
    H,W=boundary_mask.shape
    for r in range(search_radius+1):
        for s in(-1,1):
            t=q+n*(r*s)
            x=int(round(t[0])); y=int(round(t[1]))
            if 0<=x<W and 0<=y<H and boundary_mask[y,x]: return r,(x,y)
    return search_radius+1,(int(round(q[0])),int(round(q[1])))

def rolling_median(a,k):
    k=max(3,k|1)
    p=k//2
    ap=np.pad(a,(p,p),mode='edge')
    return np.array([np.median(ap[i:i+k]) for i in range(len(a))])

def precise_contact(line_a,line_b,rough_y,pre_rows=28,post_rows=140,step=1,run_len=8,sigma_k=2.5,min_abs=1.2,med_win=7):
    dvec=np.array([line_a,1.0],np.float32); dvec/=np.linalg.norm(dvec)
    nvec=np.array([-dvec[1],dvec[0]],np.float32)
    ys=np.arange(rough_y-pre_rows,rough_y+post_rows,step).astype(np.float32)
    qs=np.column_stack([x_at_y(line_a,line_b,ys),ys]).astype(np.float32)
    dists=[]; pts=[]
    for q in qs:
        r,p=nearest_boundary_distance(q,nvec)
        dists.append(float(r)); pts.append(p)
    dists=np.array(dists,np.float32)
    dists_med=rolling_median(dists,med_win)
    base_len=min(25,pre_rows//2+8)
    base=np.median(dists_med[:base_len])
    mad=np.median(np.abs(dists_med[:base_len]-base))
    thr=base+max(min_abs,1.4826*mad*sigma_k)
    idx=base_len
    cnt=0; found=-1
    while idx<len(dists_med):
        if dists_med[idx]>thr:
            cnt+=1
            if cnt>=run_len:
                found=idx-run_len+1
                break
        else:
            cnt=0
        idx+=1
    if found<0: found=np.argmax(dists_med)
    lo=max(0,found-2); hi=min(len(dists_med)-1,found+2)
    x1,y1=dists_med[lo],lo
    x2,y2=dists_med[hi],hi
    if y2!=y1:
        frac=(thr-x1)/((x2-x1)+1e-9)
        ref_idx=int(round(y1+frac*(y2-y1)))
    else:
        ref_idx=found
    ref_idx=max(0,min(len(qs)-1,ref_idx))
    q_ref=qs[ref_idx]
    _,p_ref=nearest_boundary_distance(q_ref,nvec)
    return tuple(p_ref),(ys[ref_idx],dists_med[ref_idx],thr,dists_med)

rough_y=int(row_y[break_pos])
left_contact,(yL,dl,thL,dl_series)=precise_contact(aL,bL,rough_y)
right_contact,(yR,dr,thR,dr_series)=precise_contact(aR,bR,rough_y)

plt.figure(figsize=(6,4));
plt.plot(np.arange(len(dl_series)),dl_series,label='d_med')
plt.axhline(thL,ls='--')
plt.axvline(int(yL-(rough_y-28)),ls='--')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'11_distance_profile_left.png')); plt.close()

plt.figure(figsize=(6,4));
plt.plot(np.arange(len(dr_series)),dr_series,label='d_med')
plt.axhline(thR,ls='--')
plt.axvline(int(yR-(rough_y-28)),ls='--')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'11_distance_profile_right.png')); plt.close()

final_vis=input_bgr.copy()
cv2.circle(final_vis,left_contact,6,(0,0,255),-1)
cv2.circle(final_vis,right_contact,6,(0,0,255),-1)
cv2.line(final_vis,left_contact,right_contact,(0,0,255),2)
cv2.imwrite(os.path.join(output_dir,'11_contact_points_refined.png'),final_vis)

for i,(cx,cy) in enumerate([left_contact,right_contact],1):
    x1=max(0,cx-40); y1=max(0,cy-40)
    x2=min(final_vis.shape[1],cx+41); y2=min(final_vis.shape[0],cy+41)
    cv2.imwrite(os.path.join(output_dir,f'11_contact_roi_{i}.png'),final_vis[y1:y2,x1:x2])
