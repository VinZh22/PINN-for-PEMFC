### Proofs
$x_* = (x-X_m)/X_{std}$
Same for $y_*$ and $t_*$

The equations are 
$$
\frac{du}{dt} + u \frac{du}{dx} + v \frac{du}{dy} + \frac{dp}{dx} = \nu (\frac{d^2u}{dx^2} + \frac{d^2u}{dy^2})
$$

Donc on a : 
$$
\frac{1}{T_{std}}\frac{du}{dt_*} +\frac{1}{X_{std}} u \frac{du}{dx_*} + \frac{1}{Y_{std}} v \frac{du}{dy_*} + \frac{1}{X_{std}} \frac{dp}{dx_*} = \nu (\frac{1}{X_{std}^2} \frac{d^2u}{dx_*^2} + \frac{1}{Y_{std}^2} \frac{d^2u}{dy_*^2})
$$

Posons : $\hat{U} = \frac{X_{std}}{T_{Std}}$ et de même pour $\hat{V}$ et on a $\hat{P} = \frac{\hat{U}\nu}{X_{std}} = \frac{\nu}{T_{std}}$.

On a alors en posant $u_* = \frac{u}{\hat{U}}$ etc... 
$$
\frac{\hat{U}}{T_{std}}\frac{du_*}{dt_*} +\frac{\hat{U}^2}{X_{std}} u_* \frac{du_*}{dx_*} + \frac{\hat{U}\hat{V}}{Y_{std}} v_* \frac{du_*}{dy_*} + \frac{\nu}{X_{std} T_{std}} \frac{dp_*}{dx_*} = \nu (\frac{\hat{U}}{X_{std}^2} \frac{d^2u_*}{dx_*^2} + \frac{\hat{U}^2}{Y_{std}^2} \frac{d^2u_*}{dy_*^2})
$$

Après simplification: 
$$
\frac{\hat{U}}{T_{std}}\frac{du_*}{dt_*} +\frac{\hat{U}}{T_{std}} u_* \frac{du_*}{dx_*} + \frac{\hat{U}}{T_{std}} v_* \frac{du_*}{dy_*} + \frac{\nu}{X_{std} T_{std}} \frac{dp_*}{dx_*} = \nu (\frac{\hat{U}}{X_{std}^2} \frac{d^2u_*}{dx_*^2} + \frac{\hat{U}}{Y_{std}^2} \frac{d^2u_*}{dy_*^2})
$$

Donc : 
$$
\frac{du_*}{dt_*} + u_* \frac{du_*}{dx_*} +v_* \frac{du_*}{dy_*} + \frac{\nu}{X_{std} \hat{U}} \frac{dp_*}{dx_*} = \nu (\frac{T_{std}}{X_{std}^2} \frac{d^2u_*}{dx_*^2} + \frac{T_{std}}{Y_{std}^2} \frac{d^2u_*}{dy_*^2})
$$

Ensuite : 
$$
\frac{du_*}{dt_*} + u_* \frac{du_*}{dx_*} +v_* \frac{du_*}{dy_*} + \frac{\nu}{X_{std} \hat{U}} \frac{dp_*}{dx_*} = \nu (\frac{1}{X_{std}\hat{U}} \frac{d^2u_*}{dx_*^2} + \frac{1}{Y_{std}\hat{V}} \frac{d^2u_*}{dy_*^2})
$$

Supposons alors maintenant que :
$$
X_{std} \approx Y_{std}
$$

On a ainsi en posant $Re = \frac{U L}{\nu}$ :
$$
\frac{du_*}{dt_*} + u_* \frac{du_*}{dx_*} +v_* \frac{du_*}{dy_*} = \frac{1}{Re} (\frac{d^2u_*}{dx_*^2} +  \frac{d^2u_*}{dy_*^2} - \frac{dp_*}{dx_*})
$$