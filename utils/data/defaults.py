SIZE = (256, 256)
anomalies = [
    'oscillating_tile',
    'first_order_high_noise',
    'first_order_data_loss',
    'third_order_data_loss',
    'lightning',
    'rfi_ionosphere_reflect',
    'galactic_plane',
    'source_in_sidelobes',
    'solar_storm']

percentage_comtamination = {'oscillating_tile':0.01,
                            'first_order_high_noise':0.01,
                            'first_order_data_loss':0.02,
                            'third_order_data_loss':0.04,
                            'rfi_ionosphere_reflect':0.04,
                            'lightning':0.06,
                            'galactic_plane':0.08,
                            'source_in_sidelobes':0.06,
                            'solar_storm':0.02}
