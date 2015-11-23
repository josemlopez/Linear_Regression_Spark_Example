
# **Linear Regression Example with SPARK**

#### Note that, for reference, you can look up the details of the relevant Spark methods in [Spark's Python API](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD) and the relevant NumPy methods in the [NumPy Reference](http://docs.scipy.org/doc/numpy/reference/index.html)

### ** Part 1: Read and parse the initial dataset **

#### ** (1a) Load and check the data **
#### The raw data is currently stored in text file.  We will start by storing this raw data in as an RDD, with each element of the RDD representing a data point as a comma-delimited string. Each string starts with the label (a year) followed by numerical audio features. Use the [count method](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.count) to check how many data points we have.  Then use the [take method](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.take) to create and print out a list of the first 5 data points in their initial string format.


    # load testing library
    from test_helper import Test
    import os.path
    baseDir = os.path.join('data')
    inputPath = os.path.join('cs190', 'millionsong.txt')
    fileName = os.path.join(baseDir, inputPath)
    
    numPartitions = 2
    rawData = sc.textFile(fileName, numPartitions)


    #rawData.count()


    # TODO: Replace <FILL IN> with appropriate code
    numPoints = rawData.count()
    print numPoints
    samplePoints = rawData.take(5)
    print samplePoints

    6724
    [u'2001.0,0.884123733793,0.610454259079,0.600498416968,0.474669212493,0.247232680947,0.357306088914,0.344136412234,0.339641227335,0.600858840135,0.425704689024,0.60491501652,0.419193351817', u'2001.0,0.854411946129,0.604124786151,0.593634078776,0.495885413963,0.266307830936,0.261472105188,0.506387076327,0.464453565511,0.665798573683,0.542968988766,0.58044428577,0.445219373624', u'2001.0,0.908982970575,0.632063159227,0.557428975183,0.498263761394,0.276396052336,0.312809861625,0.448530069406,0.448674249968,0.649791323916,0.489868662682,0.591908113534,0.4500023818', u'2001.0,0.842525219898,0.561826888508,0.508715259692,0.443531142139,0.296733836002,0.250213568176,0.488540873206,0.360508747659,0.575435243185,0.361005878554,0.678378718617,0.409036786173', u'2001.0,0.909303285534,0.653607720915,0.585580794716,0.473250503005,0.251417011835,0.326976795524,0.40432273022,0.371154511756,0.629401917965,0.482243251755,0.566901413923,0.463373691946']



    # TEST Load and check the data (1a)
    Test.assertEquals(numPoints, 6724, 'incorrect value for numPoints')
    Test.assertEquals(len(samplePoints), 5, 'incorrect length for samplePoints')

    1 test passed.
    1 test passed.


#### ** (1b) Using `LabeledPoint` **
#### In MLlib, labeled training instances are stored using the [LabeledPoint](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.regression.LabeledPoint) object.  Write the parsePoint function that takes as input a raw data point, parses it using Python's [unicode.split](https://docs.python.org/2/library/string.html#string.split) method, and returns a `LabeledPoint`.  Use this function to parse samplePoints (from the previous question).  Then print out the features and label for the first training point, using the `LabeledPoint.features` and `LabeledPoint.label` attributes. Finally, calculate the number features for this dataset.
#### Note that `split()` can be called directly on a `unicode` or `str` object.  For example, `u'split,me'.split(',')` returns `[u'split', u'me']`.


    from pyspark.mllib.regression import LabeledPoint
    import numpy as np
    
    # Here is a sample raw data point:
    # '2001.0,0.884,0.610,0.600,0.474,0.247,0.357,0.344,0.33,0.600,0.425,0.60,0.419'
    # In this raw data point, 2001.0 is the label, and the remaining values are features


    hg = LabeledPoint(0.0, [0.0])
    hg.features




    DenseVector([0.0])




    # TODO: Replace <FILL IN> with appropriate code
    def parsePoint(line):
        """Converts a comma separated unicode string into a `LabeledPoint`.
    
        Args:
            line (unicode): Comma separated unicode string where the first element is the label and the
                remaining elements are features.
    
        Returns:
            LabeledPoint: The line is converted into a `LabeledPoint`, which consists of a label and
                features.
        """
        labelAndFeatures = line.split(',')
        print labelAndFeatures
        return LabeledPoint(labelAndFeatures[0], labelAndFeatures[1:])
    
    parsedSamplePoints = rawData.map(parsePoint)
    firstPointFeatures = parsedSamplePoints.take(1)[0].features
    firstPointLabel = parsedSamplePoints.take(1)[0].label
    print firstPointFeatures, firstPointLabel
    
    d = len(firstPointFeatures)
    print d

    [0.884123733793,0.610454259079,0.600498416968,0.474669212493,0.247232680947,0.357306088914,0.344136412234,0.339641227335,0.600858840135,0.425704689024,0.60491501652,0.419193351817] 2001.0
    12



    # TEST Using LabeledPoint (1b)
    Test.assertTrue(isinstance(firstPointLabel, float), 'label must be a float')
    expectedX0 = [0.8841,0.6105,0.6005,0.4747,0.2472,0.3573,0.3441,0.3396,0.6009,0.4257,0.6049,0.4192]
    Test.assertTrue(np.allclose(expectedX0, firstPointFeatures, 1e-4, 1e-4),
                    'incorrect features for firstPointFeatures')
    Test.assertTrue(np.allclose(2001.0, firstPointLabel), 'incorrect label for firstPointLabel')
    Test.assertTrue(d == 12, 'incorrect number of features')

    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.


#### **Visualization 1: Features**
#### First we will load and setup the visualization library.  Then we will look at the raw features for 50 data points by generating a heatmap that visualizes each feature on a grey-scale and shows the variation of each feature across the 50 sample data points.  The features are all between 0 and 1, with values closer to 1 represented via darker shades of grey.


    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    sampleMorePoints = rawData.take(50)
    # You can uncomment the line below to see randomly selected features.  These will be randomly
    # selected each time you run the cell.  Note that you should run this cell with the line commented
    # out when answering the lab quiz questions.
    # sampleMorePoints = rawData.takeSample(False, 50)
    
    parsedSampleMorePoints = map(parsePoint, sampleMorePoints)
    dataValues = map(lambda lp: lp.features.toArray(), parsedSampleMorePoints)
    
    def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                    gridWidth=1.0):
        """Template for generating the plot layout."""
        plt.close()
        fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
        ax.axes.tick_params(labelcolor='#999999', labelsize='10')
        for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
            axis.set_ticks_position('none')
            axis.set_ticks(ticks)
            axis.label.set_color('#999999')
            if hideLabels: axis.set_ticklabels([])
        plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
        map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
        return fig, ax
    
    # generate layout and plot
    fig, ax = preparePlot(np.arange(.5, 11, 1), np.arange(.5, 49, 1), figsize=(8,7), hideLabels=True,
                          gridColor='#eeeeee', gridWidth=1.1)
    image = plt.imshow(dataValues,interpolation='nearest', aspect='auto', cmap=cm.Greys)
    for x, y, s in zip(np.arange(-.125, 12, 1), np.repeat(-.75, 12), [str(x) for x in range(12)]):
        plt.text(x, y, s, color='#999999', size='10')
    plt.text(4.7, -3, 'Feature', color='#999999', size='11'), ax.set_ylabel('Observation')
    pass

    [u'2001.0', u'0.884123733793', u'0.610454259079', u'0.600498416968', u'0.474669212493', u'0.247232680947', u'0.357306088914', u'0.344136412234', u'0.339641227335', u'0.600858840135', u'0.425704689024', u'0.60491501652', u'0.419193351817']
    [u'2001.0', u'0.854411946129', u'0.604124786151', u'0.593634078776', u'0.495885413963', u'0.266307830936', u'0.261472105188', u'0.506387076327', u'0.464453565511', u'0.665798573683', u'0.542968988766', u'0.58044428577', u'0.445219373624']
    [u'2001.0', u'0.908982970575', u'0.632063159227', u'0.557428975183', u'0.498263761394', u'0.276396052336', u'0.312809861625', u'0.448530069406', u'0.448674249968', u'0.649791323916', u'0.489868662682', u'0.591908113534', u'0.4500023818']
    [u'2001.0', u'0.842525219898', u'0.561826888508', u'0.508715259692', u'0.443531142139', u'0.296733836002', u'0.250213568176', u'0.488540873206', u'0.360508747659', u'0.575435243185', u'0.361005878554', u'0.678378718617', u'0.409036786173']
    [u'2001.0', u'0.909303285534', u'0.653607720915', u'0.585580794716', u'0.473250503005', u'0.251417011835', u'0.326976795524', u'0.40432273022', u'0.371154511756', u'0.629401917965', u'0.482243251755', u'0.566901413923', u'0.463373691946']
    [u'2001.0', u'0.8989401401', u'0.566433892669', u'0.64859417707', u'0.543599944748', u'0.225381848576', u'0.308728880753', u'0.563581838973', u'0.417239632632', u'0.57594367474', u'0.388780186721', u'0.58658505908', u'0.412239376631']
    [u'2001.0', u'0.899621729127', u'0.634814835284', u'0.544244512792', u'0.488838920978', u'0.220724942484', u'0.392549654601', u'0.406432622133', u'0.369356753691', u'0.724312625601', u'0.603798371173', u'0.521460401914', u'0.497389527096']
    [u'2001.0', u'0.843050575672', u'0.584452781625', u'0.605873877708', u'0.552002475337', u'0.250963802921', u'0.349175966461', u'0.503276444787', u'0.453911624314', u'0.598697783319', u'0.486325950149', u'0.563726897444', u'0.379721625356']
    [u'2001.0', u'0.879490939575', u'0.636515642881', u'0.559724690805', u'0.445088083563', u'0.286239972962', u'0.248357908913', u'0.472511698912', u'0.468466910891', u'0.685518543766', u'0.590246735834', u'0.530378044548', u'0.458209301646']
    [u'2007.0', u'0.767243601488', u'0.662206352254', u'0.316684955198', u'0.417922349233', u'0.297382512471', u'0.458914495662', u'0.430869709695', u'0.339996167048', u'0.655462702409', u'0.499914957625', u'0.489479446349', u'0.538481367949']
    [u'2008.0', u'0.618924286226', u'0.517881847203', u'0.327784195594', u'0.438918845319', u'0.282612577811', u'0.570471211728', u'0.502980891801', u'0.381806895805', u'0.591610802148', u'0.33687842661', u'0.258198669492', u'0.321699790029']
    [u'2002.0', u'0.582973960062', u'0.494906711937', u'0.3748120655', u'0.295112144007', u'0.186507884989', u'0.294844336884', u'0.463787626086', u'0.464679416178', u'0.546618974706', u'0.451240987501', u'0.457869272887', u'0.418605154922']
    [u'2004.0', u'0.309617027413', u'0.25749005178', u'0.384942835571', u'0.393766280475', u'0.340499471454', u'0.284685235124', u'0.490791264466', u'0.513048089201', u'0.56989618399', u'0.50841786634', u'0.519187529821', u'0.4903795845']
    [u'2003.0', u'0.58346277148', u'0.509924394818', u'0.35049347592', u'0.354847445264', u'0.278300305503', u'0.287171965487', u'0.487689567244', u'0.27580912077', u'0.563387297989', u'0.353470991249', u'0.682175571765', u'0.649439061488']
    [u'1999.0', u'0.618585576458', u'0.548511199283', u'0.289918226158', u'0.408078870746', u'0.213136211608', u'0.36661206092', u'0.460180707619', u'0.376831892188', u'0.621858352074', u'0.529513192198', u'0.565490443702', u'0.447636067947']
    [u'2003.0', u'0.518869518745', u'0.424329917638', u'0.382693311259', u'0.39665499942', u'0.29246732632', u'0.311552536138', u'0.457942036646', u'0.370190867535', u'0.58722324154', u'0.513200669443', u'0.605362175296', u'0.6184575225']
    [u'2002.0', u'0.484042011784', u'0.36570954814', u'0.195083797176', u'0.369210627918', u'0.331212890591', u'0.409598289365', u'0.30399726168', u'0.200823058526', u'0.680420746858', u'0.471064996293', u'0.540715538412', u'0.377271213364']
    [u'1992.0', u'0.674246799978', u'0.522524079376', u'0.547404737097', u'0.517455610491', u'0.195960197128', u'0.370101892266', u'0.520731075026', u'0.390485651752', u'0.532645688713', u'0.382298861495', u'0.628336012068', u'0.428213612411']
    [u'1997.0', u'0.5777096107', u'0.589552603413', u'0.558605119245', u'0.529434582861', u'0.24984801906', u'0.482406913339', u'0.469210612701', u'0.39354097987', u'0.554281437919', u'0.487293307165', u'0.588379466842', u'0.469291436908']
    [u'1987.0', u'0.633153530197', u'0.555522822196', u'0.450291561748', u'0.370419892015', u'0.300797282449', u'0.332722644823', u'0.385483136668', u'0.38306810683', u'0.612958097662', u'0.463799686672', u'0.553100770805', u'0.51421287964']
    [u'2000.0', u'0.572156340188', u'0.381727244629', u'0.414291137317', u'0.350112028519', u'0.304404368583', u'0.410370433021', u'0.453809578568', u'0.647024315412', u'0.627616498292', u'0.769165594778', u'0.47018589949', u'0.665324886404']
    [u'2000.0', u'0.674085906706', u'0.566072685897', u'0.423983451782', u'0.374007048666', u'0.30199602622', u'0.416485046636', u'0.45011202001', u'0.656449250186', u'0.645070589494', u'0.847790426355', u'0.391246389593', u'0.641219203467']
    [u'2005.0', u'0.69320277433', u'0.489076819248', u'0.350381229289', u'0.348606690315', u'0.404752945656', u'0.408309532782', u'0.353522921472', u'0.439996266348', u'0.638653697475', u'0.703257255842', u'0.449401453333', u'0.445739743114']
    [u'2000.0', u'0.474499274877', u'0.368074069798', u'0.512663206802', u'0.437986247262', u'0.308886223877', u'0.489804976988', u'0.41585630966', u'0.658393309171', u'0.668476257773', u'0.778307483995', u'0.54608477411', u'0.745070016112']
    [u'1997.0', u'0.743630306504', u'0.598082131249', u'0.39721160701', u'0.401493723689', u'0.278709233217', u'0.394315347453', u'0.477125937659', u'0.44735423171', u'0.637618839417', u'0.394268575779', u'0.528606283358', u'0.532613807425']
    [u'1997.0', u'0.700017683544', u'0.59280288173', u'0.370713341568', u'0.423296210519', u'0.232203523501', u'0.430989640311', u'0.473080180888', u'0.401516586674', u'0.579682645113', u'0.466304948669', u'0.505027205859', u'0.465325494106']
    [u'1996.0', u'0.783568136952', u'0.645079545601', u'0.396584218713', u'0.363298950898', u'0.191195712503', u'0.341917342585', u'0.463403930773', u'0.416998004416', u'0.561480664813', u'0.487650584799', u'0.491248099187', u'0.399512720873']
    [u'1997.0', u'0.724079566628', u'0.573574275736', u'0.336204689344', u'0.390894796298', u'0.263231311139', u'0.359187464528', u'0.507572986889', u'0.410279526334', u'0.648747557468', u'0.471931531587', u'0.519322743175', u'0.470028422819']
    [u'1997.0', u'0.670507012469', u'0.559718118472', u'0.270730315814', u'0.42606266875', u'0.287085544543', u'0.435735001081', u'0.468306661514', u'0.439875452239', u'0.605086702992', u'0.471080111246', u'0.507971482194', u'0.579842068007']
    [u'1997.0', u'0.60578058131', u'0.533214459774', u'0.308806654104', u'0.374687746186', u'0.213009670809', u'0.397349468725', u'0.517561457777', u'0.420226113258', u'0.58224636119', u'0.399572097331', u'0.429298180526', u'0.400720675969']
    [u'1997.0', u'0.775712326755', u'0.571582738199', u'0.438112053581', u'0.38032285825', u'0.314436268823', u'0.322984474835', u'0.490747409456', u'0.437312538154', u'0.583631378393', u'0.389085973855', u'0.485615985638', u'0.423669176346']
    [u'1997.0', u'0.693871609622', u'0.549734295891', u'0.344711925933', u'0.388232656428', u'0.28739543213', u'0.427944905942', u'0.500949439352', u'0.466914753013', u'0.556307265363', u'0.394724681955', u'0.44057972559', u'0.433430147478']
    [u'1997.0', u'0.653265679675', u'0.516383626686', u'0.244297806235', u'0.426760449412', u'0.304362915562', u'0.414005500523', u'0.506857665082', u'0.489172463077', u'0.574953418025', u'0.443579699322', u'0.559533729293', u'0.422705086252']
    [u'1997.0', u'0.680717113084', u'0.578233029493', u'0.332534888289', u'0.381705177612', u'0.25680097535', u'0.339873178384', u'0.473473146756', u'0.466324805034', u'0.580025558767', u'0.524630896161', u'0.556561477782', u'0.54185705551']
    [u'1997.0', u'0.791010186555', u'0.645351569111', u'0.376565513602', u'0.345136184234', u'0.220639693096', u'0.368781867866', u'0.4652943555', u'0.411353282409', u'0.549487535914', u'0.48991633292', u'0.398270157169', u'0.381366635935']
    [u'1998.0', u'0.769916490015', u'0.583858732883', u'0.373257956235', u'0.379598644205', u'0.257904422965', u'0.392967729', u'0.443211796834', u'0.355169315732', u'0.65357834021', u'0.500306617626', u'0.536722637002', u'0.418736202955']
    [u'2000.0', u'0.906671110853', u'0.683490813437', u'0.448109465271', u'0.384176976907', u'0.239007098082', u'0.33111427073', u'0.422129017117', u'0.392623454641', u'0.594001814223', u'0.415632316634', u'0.533873387459', u'0.457605226082']
    [u'2000.0', u'0.873902841557', u'0.656737062903', u'0.381109081566', u'0.367056534638', u'0.250450852098', u'0.344560330397', u'0.433188405254', u'0.406829097466', u'0.586859185305', u'0.448558000229', u'0.492053162605', u'0.442846492667']
    [u'2001.0', u'0.893211260469', u'0.653300261312', u'0.458460760794', u'0.396266736994', u'0.240409039764', u'0.384354139141', u'0.426200327957', u'0.41088161089', u'0.606844328485', u'0.470068239862', u'0.544877845059', u'0.567936104233']
    [u'2000.0', u'0.725028738825', u'0.557717258934', u'0.380624684197', u'0.371958688918', u'0.209119362773', u'0.357251962117', u'0.42574001847', u'0.383204698488', u'0.574938630096', u'0.471271788787', u'0.47298430526', u'0.344485779083']
    [u'2000.0', u'0.679689701629', u'0.484101845702', u'0.351215467815', u'0.510079483616', u'0.21848009579', u'0.534831777508', u'0.565192898159', u'0.278810054956', u'0.546795479621', u'0.311414879227', u'0.511840715561', u'0.373302428084']
    [u'2000.0', u'0.768918019728', u'0.545606147161', u'0.399876578615', u'0.376468032007', u'0.27048898349', u'0.374838027806', u'0.464142452985', u'0.343971227044', u'0.58109587222', u'0.38227876359', u'0.499108013711', u'0.371070004742']
    [u'2000.0', u'0.778914004754', u'0.626993529906', u'0.334723722559', u'0.356444011492', u'0.229056514715', u'0.39662785974', u'0.392466266467', u'0.412647924187', u'0.612590240513', u'0.413453271873', u'0.565372104264', u'0.568430009274']
    [u'2000.0', u'0.90437053328', u'0.627868424769', u'0.459276457914', u'0.387251538203', u'0.227787659045', u'0.334277218676', u'0.456182168236', u'0.401167825581', u'0.599695760639', u'0.44029460539', u'0.549424477347', u'0.451562902174']
    [u'1998.0', u'0.727402895637', u'0.548013366919', u'0.388713354145', u'0.442426752363', u'0.228125586005', u'0.414142491119', u'0.516705636622', u'0.476475727784', u'0.599441812114', u'0.592273634472', u'0.471285368339', u'0.365542953468']
    [u'2000.0', u'0.831155265196', u'0.717262325258', u'0.38065300788', u'0.443710363509', u'0.243676932992', u'0.33615345102', u'0.493014708673', u'0.383467289884', u'0.594212527359', u'0.501867610283', u'0.546267278835', u'0.48886464179']
    [u'2000.0', u'0.805715488209', u'0.642066000357', u'0.383282992874', u'0.392379109096', u'0.232404755154', u'0.362068920475', u'0.502577070257', u'0.397180960005', u'0.622655474877', u'0.482377957548', u'0.447700074134', u'0.477940509094']
    [u'2000.0', u'0.795244426557', u'0.549739643557', u'0.424481150052', u'0.362198705553', u'0.219524636481', u'0.346658703034', u'0.407283399721', u'0.295812075475', u'0.656442981726', u'0.473230919281', u'0.538621396614', u'0.517044183673']
    [u'2000.0', u'0.858615773378', u'0.560910502413', u'0.429686044824', u'0.407302598604', u'0.273826719265', u'0.396198682408', u'0.423495103081', u'0.393436163977', u'0.56026769833', u'0.388304647034', u'0.564209580255', u'0.482655493833']
    [u'2003.0', u'0.795184336844', u'0.592890816509', u'0.384629652995', u'0.414518962718', u'0.288667169849', u'0.31063368682', u'0.475652447748', u'0.349878319781', u'0.615836279932', u'0.387147273462', u'0.599900355085', u'0.473582402356']



![png](output_14_1.png)


#### **(1c) Find the range **
#### Now let's examine the labels to find the range of song years.  To do this, first parse each element of the `rawData` RDD, and then find the smallest and largest labels.


    # TODO: Replace <FILL IN> with appropriate code
    parsedDataInit = rawData.map(parsePoint)
    onlyLabels = parsedDataInit.map(lambda song: song.label)
    minYear = onlyLabels.reduce(lambda x,y: x if x<y else y)
    maxYear = onlyLabels.reduce(lambda x,y: x if x>y else y)
    print maxYear, minYear

    2011.0 1922.0



    # TEST Find the range (1c)
    Test.assertEquals(len(parsedDataInit.take(1)[0].features), 12,
                      'unexpected number of features in sample point')
    sumFeatTwo = parsedDataInit.map(lambda lp: lp.features[2]).sum()
    Test.assertTrue(np.allclose(sumFeatTwo, 3158.96224351), 'parsedDataInit has unexpected values')
    yearRange = maxYear - minYear
    Test.assertTrue(yearRange == 89, 'incorrect range for minYear to maxYear')

    1 test passed.
    1 test passed.
    1 test passed.


#### **(1d) Shift labels **
#### As we just saw, the labels are years in the 1900s and 2000s.  In learning problems, it is often natural to shift labels such that they start from zero.  Starting with `parsedDataInit`, create a new RDD consisting of `LabeledPoint` objects in which the labels are shifted such that smallest label equals zero.


    # TODO: Replace <FILL IN> with appropriate code
    parsedData = parsedDataInit.map(lambda tuple: LabeledPoint(tuple.label-minYear, tuple.features))
    #print parsedData.take(1)
    # Should be a LabeledPoint
    print type(parsedData.take(1)[0])
    # View the first point
    print '\n{0}'.format(parsedData.take(1))

    <class 'pyspark.mllib.regression.LabeledPoint'>
    
    [LabeledPoint(79.0, [0.884123733793,0.610454259079,0.600498416968,0.474669212493,0.247232680947,0.357306088914,0.344136412234,0.339641227335,0.600858840135,0.425704689024,0.60491501652,0.419193351817])]



    # TEST Shift labels (1d)
    oldSampleFeatures = parsedDataInit.take(1)[0].features
    newSampleFeatures = parsedData.take(1)[0].features
    Test.assertTrue(np.allclose(oldSampleFeatures, newSampleFeatures),
                    'new features do not match old features')
    sumFeatTwo = parsedData.map(lambda lp: lp.features[2]).sum()
    Test.assertTrue(np.allclose(sumFeatTwo, 3158.96224351), 'parsedData has unexpected values')
    minYearNew = parsedData.map(lambda lp: lp.label).min()
    maxYearNew = parsedData.map(lambda lp: lp.label).max()
    Test.assertTrue(minYearNew == 0, 'incorrect min year in shifted data')
    Test.assertTrue(maxYearNew == 89, 'incorrect max year in shifted data')

    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.


#### ** Visualization 2: Shifting labels **
#### We will look at the labels before and after shifting them.  Both scatter plots below visualize tuples storing i) a label value and ii) the number of training points with this label.  The first scatter plot uses the initial labels, while the second one uses the shifted labels.  Note that the two plots look the same except for the labels on the x-axis.


    # get data for plot
    oldData = (parsedDataInit
               .map(lambda lp: (lp.label, 1))
               .reduceByKey(lambda x, y: x + y)
               .collect())
    x, y = zip(*oldData)
    
    # generate layout and plot data
    fig, ax = preparePlot(np.arange(1920, 2050, 20), np.arange(0, 150, 20))
    plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
    ax.set_xlabel('Year'), ax.set_ylabel('Count')
    pass


![png](output_22_0.png)



    # get data for plot
    newData = (parsedData
               .map(lambda lp: (lp.label, 1))
               .reduceByKey(lambda x, y: x + y)
               .collect())
    x, y = zip(*newData)
    
    # generate layout and plot data
    fig, ax = preparePlot(np.arange(0, 120, 20), np.arange(0, 120, 20))
    plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
    ax.set_xlabel('Year (shifted)'), ax.set_ylabel('Count')
    pass


![png](output_23_0.png)


#### ** (1e) Training, validation, and test sets **
#### We're almost done parsing our dataset, and our final task involves split it into training, validation and test sets. Use the [randomSplit method](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.randomSplit) with the specified weights and seed to create RDDs storing each of these datasets. Next, cache each of these RDDs, as we will be accessing them multiple times in the remainder of this lab. Finally, compute the size of each dataset and verify that the sum of their sizes equals the value computed in Part (1a).


    # TODO: Replace <FILL IN> with appropriate code
    weights = [.8, .1, .1]
    seed = 42
    parsedTrainData, parsedValData, parsedTestData = parsedData.randomSplit(weights, seed)
    parsedTrainData.cache()
    parsedValData.cache()
    parsedTestData.cache()
    nTrain = parsedTrainData.count()
    nVal = parsedValData.count()
    nTest = parsedTestData.count()
    
    print nTrain, nVal, nTest, nTrain + nVal + nTest
    print parsedData.count()

    5371 682 671 6724
    6724



    # TEST Training, validation, and test sets (1e)
    Test.assertEquals(parsedTrainData.getNumPartitions(), numPartitions,
                      'parsedTrainData has wrong number of partitions')
    Test.assertEquals(parsedValData.getNumPartitions(), numPartitions,
                      'parsedValData has wrong number of partitions')
    Test.assertEquals(parsedTestData.getNumPartitions(), numPartitions,
                      'parsedTestData has wrong number of partitions')
    Test.assertEquals(len(parsedTrainData.take(1)[0].features), 12,
                      'parsedTrainData has wrong number of features')
    sumFeatTwo = (parsedTrainData
                  .map(lambda lp: lp.features[2])
                  .sum())
    sumFeatThree = (parsedValData
                    .map(lambda lp: lp.features[3])
                    .reduce(lambda x, y: x + y))
    sumFeatFour = (parsedTestData
                   .map(lambda lp: lp.features[4])
                   .reduce(lambda x, y: x + y))
    Test.assertTrue(np.allclose([sumFeatTwo, sumFeatThree, sumFeatFour],
                                2526.87757656, 297.340394298, 184.235876654),
                    'parsed Train, Val, Test data has unexpected values')
    Test.assertTrue(nTrain + nVal + nTest == 6724, 'unexpected Train, Val, Test data set size')
    Test.assertEquals(nTrain, 5371, 'unexpected value for nTrain')
    Test.assertEquals(nVal, 682, 'unexpected value for nVal')
    Test.assertEquals(nTest, 671, 'unexpected value for nTest')

    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.


### ** Part 2: Create and evaluate a baseline model **

#### **(2a) Average label **
#### A very simple yet natural baseline model is one where we always make the same prediction independent of the given data point, using the average label in the training set as the constant prediction value.  Compute this value, which is the average (shifted) song year for the training set.  Use an appropriate method in the [RDD API](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD).


    # TODO: Replace <FILL IN> with appropriate code
    averageTrainYear = (parsedTrainData
                        .map(lambda lp: lp.label)
                        .reduce(lambda x,y: x+y)/nTrain)
    print averageTrainYear

    53.9316700801



    # TEST Average label (2a)
    Test.assertTrue(np.allclose(averageTrainYear, 53.9316700801),
                    'incorrect value for averageTrainYear')

    1 test passed.


#### **(2b) Root mean squared error **
#### We naturally would like to see how well this naive baseline performs.  We will use root mean squared error ([RMSE](http://en.wikipedia.org/wiki/Root-mean-square_deviation)) for evaluation purposes.  Implement a function to compute RMSE given an RDD of (label, prediction) tuples, and test out this function on an example.


    import math
    # TODO: Replace <FILL IN> with appropriate code
    def squaredError(label, prediction):
        """Calculates the the squared error for a single prediction.
    
        Args:
            label (float): The correct value for this observation.
            prediction (float): The predicted value for this observation.
    
        Returns:
            float: The difference between the `label` and `prediction` squared.
        """
        return math.pow((prediction-label),2)
    
    def calcRMSE(labelsAndPreds):
        """Calculates the root mean squared error for an `RDD` of (label, prediction) tuples.
    
        Args:
            labelsAndPred (RDD of (float, float)): An `RDD` consisting of (label, prediction) tuples.
    
        Returns:
            float: The square root of the mean of the squared errors.
        """
        return math.sqrt(float(labelsAndPreds.map(lambda tuple: squaredError(tuple[0], tuple[1]))
                               .reduce(lambda x,y: x+y))/labelsAndPreds.count())
    
    labelsAndPreds = sc.parallelize([(3., 1.), (1., 2.), (2., 2.)])
    # RMSE = sqrt[((3-1)^2 + (1-2)^2 + (2-2)^2) / 3] = 1.291
    exampleRMSE = calcRMSE(labelsAndPreds)
    print exampleRMSE

    1.29099444874



    # TEST Root mean squared error (2b)
    Test.assertTrue(np.allclose(squaredError(3, 1), 4.), 'incorrect definition of squaredError')
    Test.assertTrue(np.allclose(exampleRMSE, 1.29099444874), 'incorrect value for exampleRMSE')

    1 test passed.
    1 test passed.


#### **(2c) Training, validation and test RMSE **
#### Now let's calculate the training, validation and test RMSE of our baseline model. To do this, first create RDDs of (label, prediction) tuples for each dataset, and then call calcRMSE. Note that each RMSE can be interpreted as the average prediction error for the given dataset (in terms of number of years).


    #print parsedTrainData.take(2)


    # TODO: Replace <FILL IN> with appropriate code
    labelsAndPredsTrain = parsedTrainData.map(lambda x: (x.label, averageTrainYear))
    rmseTrainBase = calcRMSE(labelsAndPredsTrain)
    
    labelsAndPredsVal = parsedValData.map(lambda x: (x.label, averageTrainYear))
    rmseValBase = calcRMSE(labelsAndPredsVal)
    
    labelsAndPredsTest = parsedTestData.map(lambda x: (x.label, averageTrainYear))
    rmseTestBase = calcRMSE(labelsAndPredsTest)
    
    print 'Baseline Train RMSE = {0:.3f}'.format(rmseTrainBase)
    print 'Baseline Validation RMSE = {0:.3f}'.format(rmseValBase)
    print 'Baseline Test RMSE = {0:.3f}'.format(rmseTestBase)

    Baseline Train RMSE = 21.306
    Baseline Validation RMSE = 21.586
    Baseline Test RMSE = 22.137



    # TEST Training, validation and test RMSE (2c)
    Test.assertTrue(np.allclose([rmseTrainBase, rmseValBase, rmseTestBase],
                                [21.305869, 21.586452, 22.136957]), 'incorrect RMSE value')

    1 test passed.


#### ** Visualization 3: Predicted vs. actual **
#### We will visualize predictions on the validation dataset. The scatter plots below visualize tuples storing i) the predicted value and ii) true label.  The first scatter plot represents the ideal situation where the predicted value exactly equals the true label, while the second plot uses the baseline predictor (i.e., `averageTrainYear`) for all predicted values.  Further note that the points in the scatter plots are color-coded, ranging from light yellow when the true and predicted values are equal to bright red when they drastically differ.


    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import get_cmap
    cmap = get_cmap('YlOrRd')
    norm = Normalize()
    
    actual = np.asarray(parsedValData
                        .map(lambda lp: lp.label)
                        .collect())
    error = np.asarray(parsedValData
                       .map(lambda lp: (lp.label, lp.label))
                       .map(lambda (l, p): squaredError(l, p))
                       .collect())
    clrs = cmap(np.asarray(norm(error)))[:,0:3]
    
    fig, ax = preparePlot(np.arange(0, 100, 20), np.arange(0, 100, 20))
    plt.scatter(actual, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=0.5)
    ax.set_xlabel('Predicted'), ax.set_ylabel('Actual')
    pass


![png](output_39_0.png)



    predictions = np.asarray(parsedValData
                             .map(lambda lp: averageTrainYear)
                             .collect())
    error = np.asarray(parsedValData
                       .map(lambda lp: (lp.label, averageTrainYear))
                       .map(lambda (l, p): squaredError(l, p))
                       .collect())
    norm = Normalize()
    clrs = cmap(np.asarray(norm(error)))[:,0:3]
    
    fig, ax = preparePlot(np.arange(53.0, 55.0, 0.5), np.arange(0, 100, 20))
    ax.set_xlim(53, 55)
    plt.scatter(predictions, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=0.3)
    ax.set_xlabel('Predicted'), ax.set_ylabel('Actual')




    (<matplotlib.text.Text at 0xb0e5340c>, <matplotlib.text.Text at 0xb0e71e0c>)




![png](output_40_1.png)


### ** Part 3: Train (via gradient descent) and evaluate a linear regression model **

#### ** (3a) Gradient summand **
#### Now let's see if we can do better via linear regression, training a model via gradient descent (we'll omit the intercept for now). Recall that the gradient descent update for linear regression is: $$ \scriptsize \mathbf{w}_{i+1} = \mathbf{w}_i - \alpha_i \sum_j (\mathbf{w}_i^\top\mathbf{x}_j  - y_j) \mathbf{x}_j \,.$$ where $ \scriptsize i $ is the iteration number of the gradient descent algorithm, and $ \scriptsize j $ identifies the observation.
#### First, implement a function that computes the summand for this update, i.e., the summand equals $ \scriptsize (\mathbf{w}^\top \mathbf{x} - y) \mathbf{x} \, ,$ and test out this function on two examples.  Use the `DenseVector` [dot](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.linalg.DenseVector.dot) method.


    from pyspark.mllib.linalg import DenseVector


    # TODO: Replace <FILL IN> with appropriate code
    def gradientSummand(weights, lp):
        """Calculates the gradient summand for a given weight and `LabeledPoint`.
    
        Note:
            `DenseVector` behaves similarly to a `numpy.ndarray` and they can be used interchangably
            within this function.  For example, they both implement the `dot` method.
    
        Args:
            weights (DenseVector): An array of model weights (betas).
            lp (LabeledPoint): The `LabeledPoint` for a single observation.
    
        Returns:
            DenseVector: An array of values the same length as `weights`.  The gradient summand.
        """
        return (weights.dot(lp.features)-lp.label)*lp.features
    
    exampleW = DenseVector([1, 1, 1])
    exampleLP = LabeledPoint(2.0, [3, 1, 4])
    # gradientSummand = (dot([1 1 1], [3 1 4]) - 2) * [3 1 4] = (8 - 2) * [3 1 4] = [18 6 24]
    summandOne = gradientSummand(exampleW, exampleLP)
    print summandOne
    
    exampleW = DenseVector([.24, 1.2, -1.4])
    exampleLP = LabeledPoint(3.0, [-1.4, 4.2, 2.1])
    summandTwo = gradientSummand(exampleW, exampleLP)
    print summandTwo

    [18.0,6.0,24.0]
    [1.7304,-5.1912,-2.5956]



    # TEST Gradient summand (3a)
    Test.assertTrue(np.allclose(summandOne, [18., 6., 24.]), 'incorrect value for summandOne')
    Test.assertTrue(np.allclose(summandTwo, [1.7304,-5.1912,-2.5956]), 'incorrect value for summandTwo')

    1 test passed.
    1 test passed.


#### ** (3b) Use weights to make predictions **
#### Next, implement a `getLabeledPredictions` function that takes in weights and an observation's `LabeledPoint` and returns a (label, prediction) tuple.  Note that we can predict by computing the dot product between weights and an observation's features.


    # TODO: Replace <FILL IN> with appropriate code
    def getLabeledPrediction(weights, observation):
        """Calculates predictions and returns a (label, prediction) tuple.
    
        Note:
            The labels should remain unchanged as we'll use this information to calculate prediction
            error later.
    
        Args:
            weights (np.ndarray): An array with one weight for each features in `trainData`.
            observation (LabeledPoint): A `LabeledPoint` that contain the correct label and the
                features for the data point.
    
        Returns:
            tuple: A (label, prediction) tuple.
        """
        return (observation.label, weights.dot(observation.features))
    
    weights = np.array([1.0, 1.5])
    predictionExample = sc.parallelize([LabeledPoint(2, np.array([1.0, .5])),
                                        LabeledPoint(1.5, np.array([.5, .5]))])
    labelsAndPredsExample = predictionExample.map(lambda lp: getLabeledPrediction(weights, lp))
    print labelsAndPredsExample.collect()

    [(2.0, 1.75), (1.5, 1.25)]



    # TEST Use weights to make predictions (3b)
    Test.assertEquals(labelsAndPredsExample.collect(), [(2.0, 1.75), (1.5, 1.25)],
                      'incorrect definition for getLabeledPredictions')

    1 test passed.


#### ** (3c) Gradient descent **
#### Next, implement a gradient descent function for linear regression and test out this function on an example.


    


    # TODO: Replace <FILL IN> with appropriate code
    def linregGradientDescent(trainData, numIters):
        """Calculates the weights and error for a linear regression model trained with gradient descent.
    
        Note:
            `DenseVector` behaves similarly to a `numpy.ndarray` and they can be used interchangably
            within this function.  For example, they both implement the `dot` method.
    
        Args:
            trainData (RDD of LabeledPoint): The labeled data for use in training the model.
            numIters (int): The number of iterations of gradient descent to perform.
    
        Returns:
            (np.ndarray, np.ndarray): A tuple of (weights, training errors).  Weights will be the
                final weights (one weight per feature) for the model, and training errors will contain
                an error (RMSE) for each iteration of the algorithm.
        """
        # The length of the training data
        n = trainData.count()
        # The number of features in the training data
        d = len(trainData.take(1)[0].features)
        w = np.zeros(d)
        alpha = 1.0
        # We will compute and store the training error after each iteration
        errorTrain = np.zeros(numIters)
        for i in range(numIters):
            # Use getLabeledPrediction from (3b) with trainData to obtain an RDD of (label, prediction)
            # tuples.  Note that the weights all equal 0 for the first iteration, so the predictions will
            # have large errors to start.
            labelsAndPredsTrain = trainData.map(lambda x: getLabeledPrediction(w, x))
            errorTrain[i] = calcRMSE(labelsAndPredsTrain)
    
            # Calculate the `gradient`.  Make use of the `gradientSummand` function you wrote in (3a).
            # Note that `gradient` sould be a `DenseVector` of length `d`.
            gradient = trainData.map(lambda x: gradientSummand(w, x)).sum()
            
            # Update the weights
            alpha_i = alpha / (n * np.sqrt(i+1))
            w -= alpha_i*gradient
        return w, errorTrain
    
    # create a toy dataset with n = 10, d = 3, and then run 5 iterations of gradient descent
    # note: the resulting model will not be useful; the goal here is to verify that
    # linregGradientDescent is working properly
    exampleN = 10
    exampleD = 3
    exampleData = (sc
                   .parallelize(parsedTrainData.take(exampleN))
                   .map(lambda lp: LabeledPoint(lp.label, lp.features[0:exampleD])))
    print exampleData.take(2)
    exampleNumIters = 5
    exampleWeights, exampleErrorTrain = linregGradientDescent(exampleData, exampleNumIters)
    print exampleWeights

    [LabeledPoint(79.0, [0.884123733793,0.610454259079,0.600498416968]), LabeledPoint(79.0, [0.854411946129,0.604124786151,0.593634078776])]
    [ 48.88110449  36.01144093  30.25350092]



    # TEST Gradient descent (3c)
    expectedOutput = [48.88110449,  36.01144093, 30.25350092]
    Test.assertTrue(np.allclose(exampleWeights, expectedOutput), 'value of exampleWeights is incorrect')
    expectedError = [79.72013547, 30.27835699,  9.27842641,  9.20967856,  9.19446483]
    Test.assertTrue(np.allclose(exampleErrorTrain, expectedError),
                    'value of exampleErrorTrain is incorrect')

    1 test passed.
    1 test passed.


#### ** (3d) Train the model **
#### Now let's train a linear regression model on all of our training data and evaluate its accuracy on the validation set.  Note that the test set will not be used here.  If we evaluated the model on the test set, we would bias our final results.
#### We've already done much of the required work: we computed the number of features in Part (1b); we created the training and validation datasets and computed their sizes in Part (1e); and, we wrote a function to compute RMSE in Part (2b).


    # TODO: Replace <FILL IN> with appropriate code
    numIters = 50
    weightsLR0, errorTrainLR0 = linregGradientDescent(parsedTrainData,50)
    
    labelsAndPreds = parsedValData.map(lambda x: getLabeledPrediction(weightsLR0, x))
    rmseValLR0 = calcRMSE(labelsAndPreds)
    
    print 'Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}'.format(rmseValBase,
                                                                           rmseValLR0)

    Validation RMSE:
    	Baseline = 21.586
    	LR0 = 19.192



    


    # TEST Train the model (3d)
    expectedOutput = [22.64535883, 20.064699, -0.05341901, 8.2931319, 5.79155768, -4.51008084,
                      15.23075467, 3.8465554, 9.91992022, 5.97465933, 11.36849033, 3.86452361]
    Test.assertTrue(np.allclose(weightsLR0, expectedOutput), 'incorrect value for weightsLR0')

    1 test passed.


#### ** Visualization 4: Training error **
#### We will look at the log of the training error as a function of iteration. The first scatter plot visualizes the logarithm of the training error for all 50 iterations.  The second plot shows the training error itself, focusing on the final 44 iterations.


    norm = Normalize()
    clrs = cmap(np.asarray(norm(np.log(errorTrainLR0))))[:,0:3]
    
    fig, ax = preparePlot(np.arange(0, 60, 10), np.arange(2, 6, 1))
    ax.set_ylim(2, 6)
    plt.scatter(range(0, numIters), np.log(errorTrainLR0), s=14**2, c=clrs, edgecolors='#888888', alpha=0.75)
    ax.set_xlabel('Iteration'), ax.set_ylabel(r'$\log_e(errorTrainLR0)$')
    pass


![png](output_58_0.png)



    norm = Normalize()
    clrs = cmap(np.asarray(norm(errorTrainLR0[6:])))[:,0:3]
    
    fig, ax = preparePlot(np.arange(0, 60, 10), np.arange(17, 22, 1))
    ax.set_ylim(17.8, 21.2)
    plt.scatter(range(0, numIters-6), errorTrainLR0[6:], s=14**2, c=clrs, edgecolors='#888888', alpha=0.75)
    ax.set_xticklabels(map(str, range(6, 66, 10)))
    ax.set_xlabel('Iteration'), ax.set_ylabel(r'Training Error')
    pass


![png](output_59_0.png)


### ** Part 4: Train using MLlib and perform grid search **

#### **(4a) `LinearRegressionWithSGD` **
#### We're already doing better than the baseline model, but let's see if we can do better by adding an intercept, using regularization, and (based on the previous visualization) training for more iterations.  MLlib's [LinearRegressionWithSGD](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.regression.LinearRegressionWithSGD) essentially implements the same algorithm that we implemented in Part (3b), albeit more efficiently and with various additional functionality, such as stochastic gradient approximation, including an intercept in the model and also allowing L1 or L2 regularization.  First use LinearRegressionWithSGD to train a model with L2 regularization and with an intercept.  This method returns a [LinearRegressionModel](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.regression.LinearRegressionModel).  Next, use the model's [weights](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.regression.LinearRegressionModel.weights) and [intercept](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.regression.LinearRegressionModel.intercept) attributes to print out the model's parameters.


    from pyspark.mllib.regression import LinearRegressionWithSGD
    # Values to use when training the linear regression model
    numIters = 500  # iterations
    alpha = 1.0  # step
    miniBatchFrac = 1.0  # miniBatchFraction
    reg = 1e-1  # regParam
    regType = 'l2'  # regType
    useIntercept = True  # intercept


    # TODO: Replace <FILL IN> with appropriate code
    firstModel = LinearRegressionWithSGD.train(parsedTrainData,numIters, alpha, miniBatchFrac, regParam=reg, 
                                               regType=regType, intercept=useIntercept)
    
    # weightsLR1 stores the model weights; interceptLR1 stores the model intercept
    weightsLR1 = firstModel.weights
    interceptLR1 = firstModel.intercept
    print weightsLR1, interceptLR1

    [16.682292427,14.7439059559,-0.0935105608897,6.22080088829,4.01454261926,-3.30214858535,11.0403027232,2.67190962854,7.18925791279,4.46093254586,8.14950409475,2.75135810882] 13.3335907631



    #print firstModel


    # TEST LinearRegressionWithSGD (4a)
    expectedIntercept = 13.3335907631
    expectedWeights = [16.682292427, 14.7439059559, -0.0935105608897, 6.22080088829, 4.01454261926, -3.30214858535,
                       11.0403027232, 2.67190962854, 7.18925791279, 4.46093254586, 8.14950409475, 2.75135810882]
    Test.assertTrue(np.allclose(interceptLR1, expectedIntercept), 'incorrect value for interceptLR1')
    Test.assertTrue(np.allclose(weightsLR1, expectedWeights), 'incorrect value for weightsLR1')

    1 test passed.
    1 test passed.


#### **(4b) Predict**
#### Now use the [LinearRegressionModel.predict()](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.regression.LinearRegressionModel.predict) method to make a prediction on a sample point.  Pass the `features` from a `LabeledPoint` into the `predict()` method.


    # TODO: Replace <FILL IN> with appropriate code
    samplePoint = parsedTrainData.take(1)[0]
    samplePrediction = firstModel.predict(samplePoint.features)
    print samplePrediction

    56.8013380112



    # TEST Predict (4b)
    Test.assertTrue(np.allclose(samplePrediction, 56.8013380112),
                    'incorrect value for samplePrediction')

    1 test passed.


#### ** (4c) Evaluate RMSE **
#### Next evaluate the accuracy of this model on the validation set.  Use the `predict()` method to create a `labelsAndPreds` RDD, and then use the `calcRMSE()` function from Part (2b).


    # TODO: Replace <FILL IN> with appropriate code
    labelsAndPreds = parsedValData.map(lambda x: (x.label, firstModel.predict(x.features)))
    rmseValLR1 = calcRMSE(labelsAndPreds)
    
    print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}' +
           '\n\tLR1 = {2:.3f}').format(rmseValBase, rmseValLR0, rmseValLR1)

    Validation RMSE:
    	Baseline = 21.586
    	LR0 = 19.192
    	LR1 = 19.691



    # TEST Evaluate RMSE (4c)
    Test.assertTrue(np.allclose(rmseValLR1, 19.691247), 'incorrect value for rmseValLR1')

    1 test passed.


#### ** (4d) Grid search **
#### We're already outperforming the baseline on the validation set by almost 2 years on average, but let's see if we can do better. Perform grid search to find a good regularization parameter.  Try `regParam` values `1e-10`, `1e-5`, and `1`.


    # TODO: Replace <FILL IN> with appropriate code
    bestRMSE = rmseValLR1
    bestRegParam = reg
    bestModel = firstModel
    
    numIters = 500
    alpha = 1.0
    miniBatchFrac = 1.0
    for reg in [1e-10, 1e-5, 1]:
        model = LinearRegressionWithSGD.train(parsedTrainData, numIters, alpha,
                                              miniBatchFrac, regParam=reg,
                                              regType='l2', intercept=True)
        labelsAndPreds = parsedValData.map(lambda lp: (lp.label, model.predict(lp.features)))
        rmseValGrid = calcRMSE(labelsAndPreds)
        print rmseValGrid
    
        if rmseValGrid < bestRMSE:
            bestRMSE = rmseValGrid
            bestRegParam = reg
            bestModel = model
    rmseValLRGrid = bestRMSE
    
    print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}\n\tLR1 = {2:.3f}\n' +
           '\tLRGrid = {3:.3f}').format(rmseValBase, rmseValLR0, rmseValLR1, rmseValLRGrid)

    17.0171700716
    17.0175981807
    23.8007746698
    Validation RMSE:
    	Baseline = 21.586
    	LR0 = 19.192
    	LR1 = 19.691
    	LRGrid = 17.017



    # TEST Grid search (4d)
    Test.assertTrue(np.allclose(17.017170, rmseValLRGrid), 'incorrect value for rmseValLRGrid')

    1 test passed.


#### ** Visualization 5: Best model's predictions**
#### Next, we create a visualization similar to 'Visualization 3: Predicted vs. actual' from Part 2 using the predictions from the best model from Part (4d) on the validation dataset.  Specifically, we create a color-coded scatter plot visualizing tuples storing i) the predicted value from this model and ii) true label.


    predictions = np.asarray(parsedValData
                             .map(lambda lp: bestModel.predict(lp.features))
                             .collect())
    actual = np.asarray(parsedValData
                        .map(lambda lp: lp.label)
                        .collect())
    error = np.asarray(parsedValData
                       .map(lambda lp: (lp.label, bestModel.predict(lp.features)))
                       .map(lambda (l, p): squaredError(l, p))
                       .collect())
    
    norm = Normalize()
    clrs = cmap(np.asarray(norm(error)))[:,0:3]
    
    fig, ax = preparePlot(np.arange(0, 120, 20), np.arange(0, 120, 20))
    ax.set_xlim(15, 82), ax.set_ylim(-5, 105)
    plt.scatter(predictions, actual, s=14**2, c=clrs, edgecolors='#888888', alpha=0.75, linewidths=.5)
    ax.set_xlabel('Predicted'), ax.set_ylabel(r'Actual')
    pass


![png](output_76_0.png)


#### ** (4e) Vary alpha and the number of iterations **
#### In the previous grid search, we set `alpha = 1` for all experiments.  Now let's see what happens when we vary `alpha`.  Specifically, try `1e-5` and `10` as values for `alpha` and also try training models for 500 iterations (as before) but also for 5 iterations. Evaluate all models on the validation set.  Note that if we set `alpha` too small the gradient descent will require a huge number of steps to converge to the solution, and if we use too large of an `alpha` it can cause numerical problems, like you'll see below for `alpha = 10`.


    # TODO: Replace <FILL IN> with appropriate code
    reg = bestRegParam
    modelRMSEs = []
    
    for alpha in [1e-5,10]:
        for numIters in [500,5]:
            model = LinearRegressionWithSGD.train(parsedTrainData, numIters, alpha,
                                                  miniBatchFrac, regParam=reg,
                                                  regType='l2', intercept=True)
            labelsAndPreds = parsedValData.map(lambda lp: (lp.label, model.predict(lp.features)))
            rmseVal = calcRMSE(labelsAndPreds)
            print 'alpha = {0:.0e}, numIters = {1}, RMSE = {2:.3f}'.format(alpha, numIters, rmseVal)
            modelRMSEs.append(rmseVal)

    alpha = 1e-05, numIters = 500, RMSE = 56.893
    alpha = 1e-05, numIters = 5, RMSE = 56.970
    alpha = 1e+01, numIters = 500, RMSE = 33110728225678989137251839534941311488306376280786874812632607716020409015644103457295574688229365257796099669135380709376.000
    alpha = 1e+01, numIters = 5, RMSE = 355124752.221



    # TEST Vary alpha and the number of iterations (4e)
    expectedResults = sorted([56.969705, 56.892949, 355124752.221221])
    Test.assertTrue(np.allclose(sorted(modelRMSEs)[:3], expectedResults), 'incorrect value for modelRMSEs')

    1 test passed.


#### **Visualization 6: Hyperparameter heat map **
#### Next, we perform a visualization of hyperparameter search using a larger set of hyperparameters (with precomputed results).  Specifically, we create a heat map where the brighter colors correspond to lower RMSE values.  The first plot has a large area with brighter colors.  In order to differentiate within the bright region, we generate a second plot corresponding to the hyperparameters found within that region.


    from matplotlib.colors import LinearSegmentedColormap
    
    # Saved parameters and results, to save the time required to run 36 models
    numItersParams = [10, 50, 100, 250, 500, 1000]
    regParams = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1]
    rmseVal = np.array([[  20.36769649,   20.36770128,   20.36818057,   20.41795354,  21.09778437,  301.54258421],
                        [  19.04948826,   19.0495    ,   19.05067418,   19.16517726,  19.97967727,   23.80077467],
                        [  18.40149024,   18.40150998,   18.40348326,   18.59457491,  19.82155716,   23.80077467],
                        [  17.5609346 ,   17.56096749,   17.56425511,   17.88442127,  19.71577117,   23.80077467],
                        [  17.0171705 ,   17.01721288,   17.02145207,   17.44510574,  19.69124734,   23.80077467],
                        [  16.58074813,   16.58079874,   16.58586512,   17.11466904,  19.6860931 ,   23.80077467]])
    
    numRows, numCols = len(numItersParams), len(regParams)
    rmseVal = np.array(rmseVal)
    rmseVal.shape = (numRows, numCols)
    
    fig, ax = preparePlot(np.arange(0, numCols, 1), np.arange(0, numRows, 1), figsize=(8, 7), hideLabels=True,
                          gridWidth=0.)
    ax.set_xticklabels(regParams), ax.set_yticklabels(numItersParams)
    ax.set_xlabel('Regularization Parameter'), ax.set_ylabel('Number of Iterations')
    
    colors = LinearSegmentedColormap.from_list('blue', ['#0022ff', '#000055'], gamma=.2)
    image = plt.imshow(rmseVal,interpolation='nearest', aspect='auto',
                        cmap = colors)


![png](output_81_0.png)



    # Zoom into the bottom left
    numItersParamsZoom, regParamsZoom = numItersParams[-3:], regParams[:4]
    rmseValZoom = rmseVal[-3:, :4]
    
    numRows, numCols = len(numItersParamsZoom), len(regParamsZoom)
    
    fig, ax = preparePlot(np.arange(0, numCols, 1), np.arange(0, numRows, 1), figsize=(8, 7), hideLabels=True,
                          gridWidth=0.)
    ax.set_xticklabels(regParamsZoom), ax.set_yticklabels(numItersParamsZoom)
    ax.set_xlabel('Regularization Parameter'), ax.set_ylabel('Number of Iterations')
    
    colors = LinearSegmentedColormap.from_list('blue', ['#0022ff', '#000055'], gamma=.2)
    image = plt.imshow(rmseValZoom,interpolation='nearest', aspect='auto',
                        cmap = colors)
    pass


![png](output_82_0.png)


### ** Part 5: Add interactions between features **

#### ** (5a) Add 2-way interactions **
#### So far, we've used the features as they were provided.  Now, we will add features that capture the two-way interactions between our existing features.  Write a function `twoWayInteractions` that takes in a `LabeledPoint` and generates a new `LabeledPoint` that contains the old features and the two-way interactions between them.  Note that a dataset with three features would have nine ( $ \scriptsize 3^2 $ ) two-way interactions.
#### You might want to use [itertools.product](https://docs.python.org/2/library/itertools.html#itertools.product) to generate tuples for each of the possible 2-way interactions.  Remember that you can combine two `DenseVector` or `ndarray` objects using [np.hstack](http://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack).


    # TODO: Replace <FILL IN> with appropriate code
    import itertools
    
    def twoWayInteractions(lp):
        """Creates a new `LabeledPoint` that includes two-way interactions.
    
        Note:
            For features [x, y] the two-way interactions would be [x^2, x*y, y*x, y^2] and these
            would be appended to the original [x, y] feature list.
    
        Args:
            lp (LabeledPoint): The label and features for this observation.
    
        Returns:
            LabeledPoint: The new `LabeledPoint` should have the same label as `lp`.  Its features
                should include the features from `lp` followed by the two-way interaction features.
        """
        newFeatures = list(itertools.product(lp.features,repeat=2))
        newList= [x*y for (x,y) in newFeatures]
    
        finalfeatures=np.hstack((lp.features,newList))
        return LabeledPoint(lp.label,finalfeatures)
    
    print twoWayInteractions(LabeledPoint(0.0, [2, 3]))
    
    # Transform the existing train, validation, and test sets to include two-way interactions.
    trainDataInteract = parsedTrainData.map(twoWayInteractions)
    valDataInteract = parsedValData.map(twoWayInteractions)
    testDataInteract = parsedTestData.map(twoWayInteractions)

    (0.0,[2.0,3.0,4.0,6.0,6.0,9.0])



    # TEST Add two-way interactions (5a)
    twoWayExample = twoWayInteractions(LabeledPoint(0.0, [2, 3]))
    Test.assertTrue(np.allclose(sorted(twoWayExample.features),
                                sorted([2.0, 3.0, 4.0, 6.0, 6.0, 9.0])),
                    'incorrect features generatedBy twoWayInteractions')
    twoWayPoint = twoWayInteractions(LabeledPoint(1.0, [1, 2, 3]))
    Test.assertTrue(np.allclose(sorted(twoWayPoint.features),
                                sorted([1.0,2.0,3.0,1.0,2.0,3.0,2.0,4.0,6.0,3.0,6.0,9.0])),
                    'incorrect features generated by twoWayInteractions')
    Test.assertEquals(twoWayPoint.label, 1.0, 'incorrect label generated by twoWayInteractions')
    Test.assertTrue(np.allclose(sum(trainDataInteract.take(1)[0].features), 40.821870576035529),
                    'incorrect features in trainDataInteract')
    Test.assertTrue(np.allclose(sum(valDataInteract.take(1)[0].features), 45.457719932695696),
                    'incorrect features in valDataInteract')
    Test.assertTrue(np.allclose(sum(testDataInteract.take(1)[0].features), 35.109111632783168),
                    'incorrect features in testDataInteract')

    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.


#### ** (5b) Build interaction model **
#### Now, let's build the new model.  We've done this several times now.  To implement this for the new features, we need to change a few variable names.  Remember that we should build our model from the training data and evaluate it on the validation data.
####  Note that you should re-run your hyperparameter search after changing features, as using the best hyperparameters from your prior model will not necessary lead to the best model.  For this exercise, we have already preset the hyperparameters to reasonable values.


    # TODO: Replace <FILL IN> with appropriate code
    numIters = 500
    alpha = 1.0
    miniBatchFrac = 1.0
    reg = 1e-10
    
    modelInteract = LinearRegressionWithSGD.train(trainDataInteract, numIters, alpha,
                                                  miniBatchFrac, regParam=reg,
                                                  regType='l2', intercept=True)
    labelsAndPredsInteract = valDataInteract.map(lambda lp: (lp.label, modelInteract.predict(lp.features)))
    rmseValInteract = calcRMSE(labelsAndPredsInteract)
    
    print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}\n\tLR1 = {2:.3f}\n\tLRGrid = ' +
           '{3:.3f}\n\tLRInteract = {4:.3f}').format(rmseValBase, rmseValLR0, rmseValLR1,
                                                     rmseValLRGrid, rmseValInteract)

    Validation RMSE:
    	Baseline = 21.586
    	LR0 = 19.192
    	LR1 = 19.691
    	LRGrid = 17.017
    	LRInteract = 15.690



    # TEST Build interaction model (5b)
    Test.assertTrue(np.allclose(rmseValInteract, 15.6894664683), 'incorrect value for rmseValInteract')

    1 test passed.


#### ** (5c) Evaluate interaction model on test data **
#### Our final step is to evaluate the new model on the test dataset.  Note that we haven't used the test set to evaluate any of our models.  Because of this, our evaluation provides us with an unbiased estimate for how our model will perform on new data.  If we had changed our model based on viewing its performance on the test set, our estimate of RMSE would likely be overly optimistic.
#### We'll also print the RMSE for both the baseline model and our new model.  With this information, we can see how much better our model performs than the baseline model.


    # TODO: Replace <FILL IN> with appropriate code
    labelsAndPredsTest = testDataInteract.map(lambda lp: (lp.label, modelInteract.predict(lp.features)))
    rmseTestInteract = calcRMSE(labelsAndPredsTest)
    
    print ('Test RMSE:\n\tBaseline = {0:.3f}\n\tLRInteract = {1:.3f}'
           .format(rmseTestBase, rmseTestInteract))

    Test RMSE:
    	Baseline = 22.137
    	LRInteract = 16.327



    # TEST Evaluate interaction model on test data (5c)
    Test.assertTrue(np.allclose(rmseTestInteract, 16.3272040537),
                    'incorrect value for rmseTestInteract')

    1 test passed.



    
