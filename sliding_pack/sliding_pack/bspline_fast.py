import casadi as cs
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt

class bspline_curve:
    def __init__(self, control_points):
        """
        Initialize a B-spline curve object with the given control points.

        Parameters
        ----------
        `control_points`: `np.ndarray` of shape `(n, 2)`
        """
        self.tck, _ = spi.splprep([control_points[:, 0], control_points[:, 1]], s=0, per=True)
        self.knots = self.tck[0]
        self.coeffs = self.tck[1]
        self.degree = self.tck[2]

        t = cs.MX.sym('t')
        coeffs_matrix = cs.horzcat(*self.coeffs).T
        curve = cs.bspline(t, coeffs_matrix, [self.knots.tolist()], [self.degree], 2, {})
        self.curve_func = cs.Function('curve_func', [t], [curve])
        tangent = cs.jacobian(curve, t)
        tangent /= cs.norm_2(tangent)
        normal = cs.vertcat(-tangent[1], tangent[0])
        self.tangent_func = cs.Function('tangent_func', [t], [tangent])
        self.normal_func = cs.Function('normal_func', [t], [normal])

        self.lim_surf_A = np.diag([1.0, 1.0, self.get_curvature()])

        self.t_samples = np.linspace(0, 1, 10000)
        self.pt_samples = np.array(spi.splev(self.t_samples, self.tck))
        self.psic_samples = np.arctan2(self.pt_samples[1, :], self.pt_samples[0, :])
        permute_idx = np.argsort(self.psic_samples)
        self.t_samples = self.t_samples[permute_idx]
        self.psic_samples = self.psic_samples[permute_idx]
        deduplicate_idx = np.where(np.diff(self.psic_samples) > 0)[0]
        self.t_samples = self.t_samples[deduplicate_idx]
        self.psic_samples = self.psic_samples[deduplicate_idx]
        tck_psic2t = spi.splrep(self.psic_samples, self.t_samples, s=0.1, per=True)
        psic = cs.MX.sym('psic')
        psic2t = cs.bspline(psic, cs.horzcat(*tck_psic2t[1]).T, [tck_psic2t[0].tolist()], [tck_psic2t[2]], 1, {})
        self.psic_to_t_func = cs.Function('psic_to_t_func', [psic], [psic2t])

    def psic_to_t(self, psic):
        """
        Convert the azimuth angle to the parameter of the B-spline curve.

        Parameters
        ----------
        `psic`: `float`
            The azimuth angle. Unit: rad

        Returns
        -------
        `float`
            The parameter of the B-spline curve.
        """
        # psic = cs.fmod(psic, 2 * cs.pi)
        # psic = cs.if_else(cs.le(psic, 0), psic + 2 * cs.pi, psic)
        # return psic / (2 * cs.pi)
        return self.psic_to_t_func(psic)
    
    def integrate(self, f, N=1000, M=1000):
        """
        Integrate the given function in the area enclosed by the B-spline curve, using Green's theorem.

        Parameters
        ----------
        `f`: `function`
            The integrand function, such as:
            ```
            def f(x, y):
                return np.sqrt(x ** 2 + y ** 2)
            ```

        Returns
        -------
        `float`
            The integral value.
        """
        def t_to_xy(t):
            pts = np.array(spi.splev(t, self.tck))
            return pts[0, :], pts[1, :]
        
        def t_to_dxdy(t):
            pts = np.array(spi.splev(t, self.tck, der=1))
            return pts[0, :], pts[1, :]
        
        t_samples = np.linspace(0, 1, N)
        x_samples, y_samples = t_to_xy(t_samples)
        dx_samples, dy_samples = t_to_dxdy(t_samples)

        s_samples = np.linspace(0, 1, M)
        t_grid, s_grid = np.meshgrid(t_samples, s_samples, indexing='ij')

        x_t = x_samples.reshape(-1, 1)
        y_t = y_samples.reshape(-1, 1)
        
        x_points = s_grid * x_t
        y_points = s_grid * y_t
        
        f_values = f(x_points, y_points)
        jacobian = s_grid * (x_t * dy_samples.reshape(-1, 1) - y_t * dx_samples.reshape(-1, 1))
        integrand = f_values * jacobian
        integral = np.trapz(np.trapz(integrand, s_samples, axis=1), t_samples)
        
        return integral
    
    def get_curvature(self):
        """
        Get the curvature squared of the B-spline curve.

        Returns
        -------
        `float`
            The curvature squared value.
        """
        area = self.integrate(lambda x, y: 1)
        integral = self.integrate(lambda x, y: np.sqrt(x ** 2 + y ** 2))
        c = integral / area
        return 1.0 / (c ** 2)

if __name__ == "__main__":
    # control_points = np.array([
    #     [0.061, 0.000], [0.061, 0.012], [0.061, 0.023], [0.061, 0.035], [0.041, 0.035], [0.020, 0.035], [0.000, 0.035], [-0.020, 0.035], [-0.041, 0.035], [-0.061, 0.035], [-0.061, 0.023], [-0.061, 0.012], [-0.061, 0.000], [-0.061, -0.012], [-0.061, -0.023], [-0.061, -0.035], [-0.041, -0.035], [-0.020, -0.035], [0.000, -0.035], [0.020, -0.035], [0.041, -0.035], [0.061, -0.035], [0.061, -0.023], [0.061, -0.012]
    # ])
    # control_points = np.array([
    #     [0.061, 0.035], [0.0, 0.035], [-0.061, 0.035], [-0.061, 0.0], [-0.061, -0.035], [0.0, -0.035], [0.061, -0.035], [0.061, 0.0]
    # ])
    # control_points = np.vstack([control_points, control_points[0]])
    # control_points = np.array([[-0.024475433234649364, 0.0017366617757171919], [-0.027199992040231692, -0.008046229310816165], [-0.026660363364003497, -0.02066711121500004], [-0.025242589876826996, -0.03072439353361952], [-0.022350935641789534, -0.04267589375497794], [-0.013540136966065307, -0.05005060019027988], [0.01078887926685118, -0.05167773517060604], [0.019782249617289867, -0.05054463565117704], [0.029603260352220464, -0.043530435964096524], [0.03049381339045585, -0.029867349792103048], [0.029082798361120592, -0.015015731566558312], [0.029644042246522864, -0.005011671125605825], [0.02941747025534378, 0.002697477868758407], [0.030192861804351, 0.021994328358193996], [0.030370004693364094, 0.03293663647307625], [0.028435563777717394, 0.041257287390087334], [0.019153225638046004, 0.04756259081300271], [0.009223959293552032, 0.049019352454667216], [-0.004444915881991857, 0.04947628922757106], [-0.012566706519195553, 0.047086089465105084], [-0.020829479777222756, 0.03908152456099308], [-0.02342233838245004, 0.02446862280130096], [-0.02343610356731351, 0.015385428433765207], [-0.024475433234649364, 0.0017366617757171919]])
    # control_points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]])
    # control_points = np.array([[-0.0855486360136761, 0.00526750272583004], [-0.08339693890826967, -0.00357354974752482], [-0.06837775176883033, -0.009790851990332115], [-0.0641622331225447, -0.014955843609600524], [-0.05230597498933223, -0.016082203193425855], [-0.04662430831825595, -0.019121364384425354], [-0.04215963172372375, -0.021779908914342117], [-0.03450207068143267, -0.027589244793100105], [-0.02992729596102204, -0.029863505133604515], [-0.012563021039217907, -0.03242967699805691], [-0.006455890084673066, -0.03037535533062742], [0.001842993980354194, -0.027098811196725567], [0.007296253025808792, -0.02637761892205412], [0.014709444485388272, -0.02654501823182643], [0.02641805504616309, -0.03027605033330394], [0.04356202285080967, -0.03284843542432448], [0.059083739248510576, -0.030592862823618776], [0.06154183990698, -0.020886215606117343], [0.06206659003819289, -0.00974664593839871], [0.06277595098101178, -0.0032857797104754566], [0.0624625953148227, 0.01194616658281625], [0.06386537614019423, 0.0274201131510053], [0.05150344301965983, 0.03654545854446863], [0.03225269921841956, 0.03575023517233661], [0.015480146261131178, 0.030799285431833634], [0.006706713056096578, 0.02976747103522339], [-0.0009790279752439801, 0.031421180824449696], [-0.007897587442323523, 0.034460783638424156], [-0.014822310029077253, 0.035712472167741864], [-0.020991732699327297, 0.03585178559331905], [-0.027658815731209506, 0.03434695974574943], [-0.0352642321725989, 0.031207945584002833], [-0.04300740665521588, 0.026229840870264302], [-0.0479633420146134, 0.023847083478236143], [-0.05449912784943931, 0.021499586054792123], [-0.06737727822777563, 0.019248479795004533], [-0.07467122368586708, 0.012821860300931685], [-0.0855486360136761, 0.00526750272583004]])
    control_points = np.array([[-0.066022005822505, -2.42817240530035e-05], [-0.06918484771295176, -0.014045859246701802], [-0.06214978376388761, -0.016427603367335984], [-0.051906611247334936, -0.017705355488195323], [-0.046027356664699426, -0.018618315247742493], [-0.03975840799810978, -0.023732125539668986], [-0.030730923610165068, -0.023527280813630577], [-0.025523766302996558, -0.02295895059957466], [-0.019327785435919127, -0.023268529580739956], [-0.012618111766193032, -0.024466799058496913], [-0.0036166174238683934, -0.024827557105739508], [0.00302561492936826, -0.025125061289085027], [0.02153309024661254, -0.026049777726332025], [0.03214058452974129, -0.02570203622458057], [0.041402670693913, -0.020346749659437478], [0.042795665958611825, -0.011135721513848836], [0.04411641374889001, -0.000537115947286477], [0.043039003889152216, 0.0182834975331322], [0.03402318604004742, 0.026057036055335343], [0.015575175069764633, 0.026074959783008906], [0.00870157573544678, 0.025045925777172552], [0.00048317133311213695, 0.02311331290871954], [-0.005450265540876313, 0.02166908055501393], [-0.01462752915281961, 0.020587773572836043], [-0.02059835952536652, 0.020015756295942166], [-0.027180846574530425, 0.02191639220369422], [-0.0347325893058717, 0.023374093138122796], [-0.04002627683471387, 0.018994030029416274], [-0.04477414312756616, 0.01379226474744558], [-0.05270394938543535, 0.008432130804835028], [-0.06241183957263892, 0.0054652852314104735], [-0.066022005822505, -2.42817240530035e-05]])
    # control_points = np.array([[-0.08767596, -0.00100248],
    #                            [-0.06916022, -0.00884441],
    #                            [-0.0548078 , -0.01586488],
    #                            [-0.0451609 , -0.02070009],
    #                            [-0.03919713, -0.0263606 ],
    #                            [-0.03234792, -0.02999773],
    #                            [-0.02281923, -0.03216117],
    #                            [-0.0149374 , -0.03260011],
    #                            [-0.009702  , -0.03079729],
    #                            [-0.00276861, -0.02814642],
    #                            [ 0.00306315, -0.02676488],
    #                            [ 0.0163923 , -0.02724202],
    #                            [ 0.03924456, -0.03383189],
    #                            [ 0.06189926, -0.02452616],
    #                            [ 0.0626202 , -0.00698884],
    #                            [ 0.06281821,  0.0131388 ],
    #                            [ 0.06316298,  0.02871977],
    #                            [ 0.04439155,  0.03593754],
    #                            [ 0.01486481,  0.03067183],
    #                            [ 0.00262043,  0.02993451],
    #                            [-0.0036024 ,  0.03148433],
    #                            [-0.01007318,  0.03405805],
    #                            [-0.01654122,  0.03576787],
    #                            [-0.02380141,  0.03561397],
    #                            [-0.03121467,  0.03385778],
    #                            [-0.03862106,  0.02993219],
    #                            [-0.04649112,  0.02600531],
    #                            [-0.05625142,  0.02124848],
    #                            [-0.0699543 ,  0.01463715],
    #                            [-0.08767596, -0.00100248]])
    curve = bspline_curve(control_points)
    print(f"{curve.pt_samples.shape=}")

    t_vals = np.linspace(0, 1, 1000)
    pt_vals = np.array([curve.curve_func(t) for t in t_vals]).reshape(-1, 2)
    print(f"{pt_vals.shape=}")

    psic = -2.527722210361019
    t = curve.psic_to_t(psic)
    print(f"{t=}")
    pt = np.array(curve.curve_func(t)).reshape(-1)
    print(f"{pt=}")
    print(f"psic: {np.arctan2(pt[1], pt[0])}, error: {np.arctan2(pt[1], pt[0]) - psic}")
    tangent = np.array(curve.tangent_func(t)).reshape(-1)
    normal = np.array(curve.normal_func(t)).reshape(-1)

    t_val = 0.5
    pt_val = curve.curve_func(t_val)
    pt_val = np.array(pt_val)
    print(f"{pt_val=}")
    print(f"{type(pt_val)=}")
    tangent_val = curve.tangent_func(t_val)
    normal_val = curve.normal_func(t_val)
    tangent_val = np.array(tangent_val)
    normal_val = np.array(normal_val)
    print(f"{tangent_val=}")
    print(f"{normal_val=}")
    print(f"{type(tangent_val)=}")
    print(f"{type(normal_val)=}")

    plt.plot(0.0, 0.0, 'ro')
    plt.plot(pt_vals[:, 0], pt_vals[:, 1], 'b-')
    plt.plot(control_points[:, 0], control_points[:, 1], 'ro')
    plt.plot(pt[0], pt[1], 'go')
    plt.quiver(pt[0], pt[1], tangent[0], tangent[1], color='r')
    plt.quiver(pt[0], pt[1], normal[0], normal[1], color='b')
    plt.axis('equal')
    plt.show()
