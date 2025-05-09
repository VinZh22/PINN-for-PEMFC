### Proofs
On remarque que $X_{std} \approx Y_{std}$, on peut alors poser $L = X_{std}$ pour représenter les deux axes.
On pose $x_* = (x-X_m)/L$ avec $X_m$ la moyenne de $x$ sur les données.
La même chose pour $y_*$
Nous allons également poser $\hat{U}$ avec $\hat{U}$ l'écart-type de $u$. On pose alors ensuite $T_{std} = L / \hat{U}$ et $t_* = (t-t_m) / T_{std}$

L'équation est : 
$$
\frac{du}{dt} + u \frac{du}{dx} + v \frac{du}{dy} + \frac{dp}{dx} = \nu (\frac{d^2u}{dx^2} + \frac{d^2u}{dy^2})
$$

Donc on a : 
$$
\frac{1}{T_{std}}\frac{du}{dt_*} +\frac{1}{L} u \frac{du}{dx_*} + \frac{1}{L} v \frac{du}{dy_*} + \frac{1}{L} \frac{dp}{dx_*} = \nu (\frac{1}{L^2} \frac{d^2u}{dx_*^2} + \frac{1}{L^2} \frac{d^2u}{dy_*^2})
$$

Posons : $u_* = \frac{u}{\hat{U}}$ et de même pour $v_*$ et on a $\hat{P} = \hat{U}^2$, pour donner $p_* = p/\hat{P}$

On a alors :
$$
\frac{\hat{U}}{T_{std}}\frac{du_*}{dt_*} +\frac{\hat{U}^2}{L} u_* \frac{du_*}{dx_*} + \frac{\hat{U}^2}{L} v_* \frac{du_*}{dy_*} + \frac{\hat{U}^2}{L} \frac{dp_*}{dx_*} = \nu (\frac{\hat{U}}{L^2} \frac{d^2u_*}{dx_*^2} + \frac{\hat{U}}{L^2} \frac{d^2u_*}{dy_*^2})
$$

Après simplification: 
$$
\frac{\hat{U}^2}{L}\frac{du_*}{dt_*} +\frac{\hat{U}^2}{L} u_* \frac{du_*}{dx_*} + \frac{\hat{U}^2}{L} v_* \frac{du_*}{dy_*} + \frac{\hat{U}^2}{L} \frac{dp_*}{dx_*} = \nu (\frac{\hat{U}}{L^2} \frac{d^2u_*}{dx_*^2} + \frac{\hat{U}}{L^2} \frac{d^2u_*}{dy_*^2})
$$

Donc : 
$$
\frac{du_*}{dt_*} + u_* \frac{du_*}{dx_*} +v_* \frac{du_*}{dy_*} +  \frac{dp_*}{dx_*} = \nu \frac{1}{L \hat{U}}( \frac{d^2u_*}{dx_*^2} + \frac{d^2u_*}{dy_*^2})
$$

On a ainsi en posant $Re = \frac{\hat{U} L}{\nu}$ :
$$
\frac{du_*}{dt_*} + u_* \frac{du_*}{dx_*} +v_* \frac{du_*}{dy_*} + \frac{dp_*}{dx_*} = \frac{1}{Re} (\frac{d^2u_*}{dx_*^2} +  \frac{d^2u_*}{dy_*^2})
$$