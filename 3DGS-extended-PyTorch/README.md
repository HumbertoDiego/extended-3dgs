# Pure Pytorch implementation of L1 loss of modified 3DGS to account curvatures

```math
G(\mathbf{p}) = \sigma_{\mu} exp(-0,5(d_x^2 x + 2d_x d_y y + d_y^2z))
```

Onde:

```math
\mathbf{p} = [p_x, p_y]; \\
d_x = \mu'_x - p_x; \\
d_y = \mu'_y - p_y; \\
\Sigma'^{-1} = cov2D^{-1} = 
\begin{bmatrix}
x & y \\
y & z
\end{bmatrix}; \\
\sigma_{\mu} = \frac{\min\{ s^2_x, s^2_y, s^2_z\}}{s^2_x + s^2_y + s^2_z}
```

## Major changes inside tile rendering loop

```python
sorted_scales = scales[in_mask][index]
scales2=sorted_scales * sorted_scales
mins = scales2.min(dim=1).values
curvatures = mins / scales2.sum(dim=1)

power = -0.5 * ( dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0])
gauss_weight = curvatures[None] * torch.exp(power)
```

## Introducing another form factor

```math
f(s) = \frac{1}{3}\left[ (1-\frac{min}{mid})+ (1-\frac{min}{max}) + (\frac{mid}{max})\right]
```

```python
sorted_scales = scales[in_mask][index]
s, i = torch.sort(sorted_scales)
# fs=1 if the gaussian assume a disk format
# fs=0 otherwise
weights = (1.0/3.0)*((1-s[:,0]/s[:,1])+(1-s[:,0]/s[:,2])+(s[:,1]/s[:,2])) # fs

gauss_weight = weights[None] * torch.exp(-0.5 * ( dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                                                + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                                                + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                                                + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
```

