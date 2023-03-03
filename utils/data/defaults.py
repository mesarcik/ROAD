SIZE = (256, 256)
anomalies = [
    'oscillating_tile',
    #'real_high_noise',
    'first_order_high_noise',
    'third_order_high_noise',
    #'electric_fence',
    'first_order_data_loss',
    'third_order_data_loss',
    'lightning',
    #'strong_radio_emitter',
    'galactic_plane',
    'source_in_sidelobes',
    #'ionosphere',
    'solar_storm']
percentage_comtamination = {'oscillating_tile':0.02,
                            #'real_high_noise':0.13,
                            'first_order_high_noise':0.05,
                            'third_order_high_noise':0.05,
                            #'electric_fence':0.01,
                            'data_loss':0.1,
                            'first_order_data_loss':0.05,
                            'third_order_data_loss':0.05,
                            'lightning':0.07,
                            #'strong_radio_emitter':0.1,
                            'galactic_plane':0.1,
                            'source_in_sidelobes':0.1,
                            #'ionosphere':0.1,
                            'solar_storm':0.07}
default_stations = [
    'CS001HBA0',
    'CS001HBA1',
    'CS001LBA',
    'CS002HBA0',
    'CS002HBA1',
    'CS002LBA',
    'CS003HBA0',
    'CS003HBA1',
    'CS003LBA',
    'CS004HBA0',
    'CS004HBA1',
    'CS004LBA',
    'CS005HBA0',
    'CS005HBA1',
    'CS005LBA',
    'CS006HBA0',
    'CS006HBA1',
    'CS006LBA',
    'CS007HBA0',
    'CS007HBA1',
    'CS007LBA',
    'CS011HBA0',
    'CS011HBA1',
    'CS011LBA',
    'CS013HBA0',
    'CS013HBA1',
    'CS013LBA',
    'CS017HBA0',
    'CS017HBA1',
    'CS017LBA',
    'CS021HBA0',
    'CS021HBA1',
    'CS021LBA',
    'CS024HBA0',
    'CS024HBA1',
    'CS024LBA',
    'CS026HBA0',
    'CS026HBA1',
    'CS026LBA',
    'CS028HBA0',
    'CS028HBA1',
    'CS028LBA',
    'CS030HBA0',
    'CS030HBA1',
    'CS030LBA',
    'CS031HBA0',
    'CS031HBA1',
    'CS031LBA',
    'CS032HBA0',
    'CS032HBA1',
    'CS032LBA',
    'CS101HBA0',
    'CS101HBA1',
    'CS101LBA',
    'CS103HBA0',
    'CS103HBA1',
    'CS103LBA',
    'CS201HBA0',
    'CS201HBA1',
    'CS201LBA',
    'CS301HBA0',
    'CS301HBA1',
    'CS301LBA',
    'CS302HBA0',
    'CS302HBA1',
    'CS302LBA',
    'CS401HBA0',
    'CS401HBA1',
    'CS401LBA',
    'CS501HBA0',
    'CS501HBA1',
    'CS501LBA',
    'DE601HBA',
    'DE601LBA',
    'DE602HBA',
    'DE602LBA',
    'DE603HBA',
    'DE603LBA',
    'DE604HBA',
    'DE604LBA',
    'DE605HBA',
    'DE605LBA',
    'DE609HBA',
    'DE609LBA',
    'FR606HBA',
    'FR606LBA',
    'IE613HBA',
    'IE613LBA',
    'LV614LBA',
    'LV614HBA',
    'PL610HBA',
    'PL610LBA',
    'PL611HBA',
    'PL611LBA',
    'PL612HBA',
    'PL612LBA',
    'RS106HBA',
    'RS106LBA',
    'RS205HBA',
    'RS205LBA',
    'RS208HBA',
    'RS208LBA',
    'RS210HBA',
    'RS210LBA',
    'RS305HBA',
    'RS305LBA',
    'RS306HBA',
    'RS306LBA',
    'RS307HBA',
    'RS307LBA',
    'RS310HBA',
    'RS310LBA',
    'RS406HBA',
    'RS406LBA',
    'RS407HBA',
    'RS407LBA',
    'RS409HBA',
    'RS409LBA',
    'RS503HBA',
    'RS503LBA',
    'RS508HBA',
    'RS508LBA',
    'RS509HBA',
    'RS509LBA',
    'SE607HBA',
    'SE607LBA',
    'UK608HBA',
    'UK608LBA']
frequency_bands = { 
        32: ['119942100.0-127250290.0', '120312500.0-126030730.0',
       '120312500.0-126054760.0', '120312500.0-126226040.0',
       '124738310.0-130657200.0', '124757000.0-130699540.0',
       '125600050.0-132908250.0', '126215360.0-132329560.0',
       '126215360.0-132524870.0', '126240160.0-132378010.0',
       '126410670.0-132524870.0', '127486040.0-134794240.0',
       '130884936.0-137413400.0', '131019590.0-137541970.0',
       '132514190.0-138823700.0', '132563400.0-138896560.0',
       '132709500.0-139019000.0', '133144000.0-140452200.0',
       '135029980.0-142338180.0', '137598800.0-143931970.0',
       '137726600.0-144036100.0', '139008340.0-145317840.0',
       '139081950.0-145415120.0', '139203650.0-145513150.0',
       '140687940.0-147996140.0', '142573940.0-149882130.0',
       '144117360.0-150255200.0', '144220740.0-150334930.0',
       '14424133.0-20195008.0', '145502460.0-151616670.0',
       '145502460.0-151811980.0', '145600510.0-151738350.0',
       '145697800.0-151811980.0', '148231890.0-155540080.0',
       '150117870.0-157426060.0', '150440590.0-156383140.0',
       '150519570.0-156438450.0', '151801300.0-157915500.0',
       '151923740.0-157866290.0', '151996600.0-157915500.0',
       '155775840.0-163084030.0', '156568530.0-162511060.0',
       '156623070.0-162541970.0', '157661820.0-164970020.0',
       '158051680.0-163994220.0', '158100130.0-164019000.0',
       '15913391.0-21684266.0', '162696460.0-174893180.0',
       '162726600.0-174945070.0', '163319780.0-170627970.0',
       '164296720.0-187304690.0', '164344780.0-187304690.0',
       '165205760.0-172513970.0', '170863730.0-178171920.0',
       '172749710.0-180057900.0', '176376340.0-124571610.0',
       '176422110.0-124553680.0', '178407660.0-125364300.0',
       '20381164.0-26152038.0', '21870422.0-27641296.0',
       '26338196.0-34140016.0', '27827454.0-37118532.0',
       '30078124.0-35844420.0', '30273438.0-36039732.0',
       '32226562.0-37992860.0', '34375000.0-38170624.0',
       '34512330.0-46054076.0', '34541320.0-40312196.0',
       '34736630.0-40507508.0', '36030580.0-41801452.0',
       '36225892.0-41996764.0', '37490844.0-49032590.0',
       '38179016.0-43949892.0', '38294220.0-42125700.0',
       '40498350.0-46269228.0', '40693664.0-46464540.0',
       '41987610.0-47758484.0', '42182924.0-47953796.0',
       '42249300.0-46080780.0', '42382812.0-45193100.0',
       '42382812.0-45217132.0', '44136050.0-49906920.0',
       '44565584.0-47451020.0', '45285416.0-48147200.0',
       '45310212.0-48195650.0', '46204376.0-50035860.0',
       '46426390.0-57968140.0', '46455384.0-52226256.0',
       '46650696.0-52421570.0', '47544096.0-50429536.0',
       '47944640.0-53715516.0', '48139950.0-53910828.0',
       '48239516.0-51101304.0', '48288730.0-51174164.0',
       '49404908.0-60946656.0', '50093080.0-55863950.0',
       '50159456.0-53990936.0', '50522612.0-53408052.0',
       '51193620.0-54055404.0', '51193620.0-54250716.0',
       '51267244.0-54152680.0', '52412416.0-58183290.0',
       '52607730.0-58378600.0', '53501130.0-56386570.0',
       '53901670.0-59672548.0', '54096984.0-59867860.0',
       '54114532.0-57946016.0', '54147720.0-57009508.0',
       '54245760.0-57131196.0', '54343030.0-57204820.0',
       '56050108.0-61820984.0', '56479644.0-59365080.0',
       '57101820.0-59963610.0', '57224270.0-60109710.0',
       '57297136.0-60158920.0', '58069612.0-61901092.0',
       '58340456.0-69882200.0', '58369444.0-64140320.0',
       '58564760.0-64335630.0', '59458160.0-62343596.0',
       '59858704.0-65629576.0', '60054016.0-65824892.0',
       '60055924.0-62917708.0', '60202788.0-63088228.0',
       '60251236.0-63113020.0', '61318970.0-72860720.0',
       '62007140.0-67778020.0', '62024690.0-65820310.0',
       '62436676.0-65322110.0', '63010024.0-66015624.0',
       '63181304.0-66015624.0', '63205336.0-66015624.0',
       '64326476.0-70097350.0', '64521788.0-70292664.0',
       '65415190.0-44472504.0', '65815736.0-71586610.0',
       '66011050.0-71781920.0', '67964180.0-73735050.0',
       '70254520.0-81796264.0', '70283510.0-76054380.0',
       '70478824.0-76249700.0', '71772770.0-77539064.0',
       '71968080.0-77734376.0', '73233030.0-84765624.0',
       '73921200.0-79687500.0', '76240536.0-34355164.0',
       '76435860.0-34550476.0', '82168580.0-14237976.0',
       '9960938.0-15727234.0'],
       64:['119942100.0-134794240.0', '120312500.0-132035820.0',
       '120312500.0-132329560.0', '120312500.0-132378010.0',
       '120312500.0-132524870.0', '126215360.0-139019000.0',
       '126240160.0-138896560.0', '127486040.0-142338180.0',
       '129364776.0-142559060.0', '129401780.0-142448800.0',
       '131258010.0-146110160.0', '132221980.0-143949890.0',
       '132514190.0-145317840.0', '132563400.0-145415120.0',
       '132709500.0-145513150.0', '135029980.0-149882130.0',
       '139081950.0-151738350.0', '139203650.0-151811980.0',
       '142573940.0-157426060.0', '142634200.0-154899980.0',
       '142743680.0-154961390.0', '144136050.0-155863950.0',
       '145502460.0-157915500.0', '145600510.0-157866290.0',
       '145697800.0-157915500.0', '146345900.0-161198050.0',
       '150117870.0-164970020.0', '151923740.0-163994220.0',
       '151996600.0-164019000.0', '155085380.0-167155840.0',
       '155146030.0-167168430.0', '156050110.0-167773440.0',
       '157661820.0-172513970.0', '158051680.0-187304690.0',
       '158100130.0-187304690.0', '15913391.0-27641296.0',
       '161433800.0-176285940.0', '164296720.0-126054760.0',
       '164344780.0-126030730.0', '165205760.0-180057900.0',
       '167341230.0-129216380.0', '167353060.0-129180140.0',
       '172749710.0-127250290.0', '176521680.0-131022264.0',
       '18891906.0-31161500.0', '21870422.0-37118532.0',
       '27827454.0-49032590.0', '30078124.0-41801452.0',
       '30273438.0-41996764.0', '31533814.0-54989624.0',
       '32226562.0-43949892.0', '34375000.0-42125700.0',
       '36030580.0-47758484.0', '36225892.0-47953796.0',
       '37490844.0-60946656.0', '39009096.0-50737000.0',
       '39204410.0-50932310.0', '41987610.0-53715516.0',
       '42182924.0-53910828.0', '42249300.0-50035860.0',
       '42382812.0-48147200.0', '42382812.0-48195650.0',
       '44136050.0-55863950.0', '45310212.0-51174164.0',
       '46799468.0-52663420.0', '47944640.0-59672548.0',
       '48139950.0-59867860.0', '48239516.0-54055404.0',
       '48239516.0-54250716.0', '48288730.0-54152680.0',
       '49404908.0-72860720.0', '50159456.0-57946016.0',
       '50923156.0-62651064.0', '51118468.0-62846376.0',
       '51267244.0-57131196.0', '52756500.0-58620452.0',
       '53901670.0-65629576.0', '54096984.0-65824892.0',
       '54147720.0-59963610.0', '54245760.0-60109710.0',
       '54343030.0-60158920.0', '55361940.0-78817750.0',
       '56050108.0-67778020.0', '57224270.0-63088228.0',
       '58069612.0-65820310.0', '58713532.0-64577484.0',
       '59858704.0-71586610.0', '60054016.0-71781920.0',
       '60055924.0-66015624.0', '60202788.0-66015624.0',
       '60251236.0-66015624.0', '61318970.0-84765624.0',
       '62837220.0-74565130.0', '63032532.0-74760440.0',
       '63181304.0-45217132.0', '64670564.0-46706390.0',
       '65815736.0-77539064.0', '66011050.0-77734376.0',
       '67964180.0-79687500.0', '71772770.0-35844420.0',
       '71968080.0-36039732.0', '73233030.0-15727234.0',
       '74751280.0-38822936.0', '74946590.0-39018250.0',
       '79190060.0-18705750.0', '9960938.0-21684266.0'] 
       }
