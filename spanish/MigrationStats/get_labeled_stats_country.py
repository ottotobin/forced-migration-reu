import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
'''
Description:

This script will count all the emotions for every labeled tweet csv file
within the labeled tweets directory. This information is then plotted as 
a time series so that we can visualize emotions as a signal and compare it
to other signals such as migration data.

How to run: 

python3 get_labeled_stats_country.py --input_dir ./LabeledTweets2022 --output_dir ./FULLLabelResults2022
python3 get_labeled_stats_country.py --input_dir ./LabeledTweets2022 --output_dir ./FULLLabelResultsNormalize2022NEW --normalize True
python3 get_labeled_stats_country.py --input_dir ./LabeledTweets2022 --output_dir ./FULLLabelResultsCOL2022 --country COL
python3 get_labeled_stats_country.py --input_dir ./LabeledTweets2022 --output_dir ./FULLLabelResultsNormalizeCOL2022 --normalize True --country COL
python3 get_labeled_stats_country.py --input_dir ./LabeledTweets2022 --output_dir ./FULLLabelResultsVEN2022 --country VEN
python3 get_labeled_stats_country.py --input_dir ./LabeledTweets2022 --output_dir ./FULLLabelResultsNormalizeVEN2022 --normalize True --country VEN

'''

# get the emotion counts from the csv files
def process_files(directory, normalize):
    emotions = ['anger', 'fear', 'joy', 'sadness', 'others']
    emotion_counts = {emotion: {} for emotion in emotions}
    # loop through every csv file within the labeled tweets dir
    for filename in glob.glob(os.path.join(directory, '*.csv')):
        date_str = os.path.basename(filename).split('_')[0]
        date = pd.to_datetime(date_str)

        #if date > pd.to_datetime('2022-10-31'):

        # barcelona excluded, rubio excluded
        city_list = []

        if args.country == 'VEN':
            city_list = ['Caracas', 'Maracaibo', 'Valencia', 'Barquisimeto', 'Maracay', 'Ciudad Guayana', 'Maturin', 'Guarenas', 'San Cristóbal', 'Ciudad Bolívar', 'Cumana', 'Barinas', 'Catia La Mar', 'Cabimas', 'Acarigua', 'Punto Fijo', 'Porlamar', 'Mérida', 'Los Teques', 'Coro', 'Santa Teresa del Tuy', 'Las Delicias', 'Guanare', 'Puerto Cabello', 'Ciudad Ojeda', 'Carora', 'Valera', 'Charallave', 'Calabozo', 'La Victoria', 'Carúpano', 'Biruaca', 'San Carlos', 'Valle de la Pascua', 'Anaco', 'Puerto Ayacucho', 'San Juan de Los Morros', 'El Vigía', 'San Fernando de Apure', 'Tucupita', 'El Tocuyo', 'Tinaquillo', 'Machiques', 'Santa Bárbara del Zulia', 'Cúa', 'Ocumare del Tuy', 'Upata', 'Cabudare', 'San Mateo', 'Socopó', 'San Félix', 'Yaritagua', 'La Concepción', 'San José de Guanipa', 'Villa del Rosario', 'San Rafael del Mojan', 'Quíbor', 'Mariara', 'Los Tanques', 'Zaraza', 'Punta de Mata', 'Marín', 'Boconó', 'Güigüe', 'Duaca', 'El Charal', 'Santa Rita', 'Ciudad Bolivia', 'Altagracia de Orituco', 'Zuata', 'Paya' 'San Juan de Colón', 'Sabana de Parra', 'Santa Cruz', 'Achaguas', 'Los Pijiguaos']

        elif args.country == 'COL':
            # armenia excluded, florencia excluded, garzón excluded, turbo excluded, manuela excluded, granada excluded, pereira excluded, el esfuerzo excluded
            city_list = ['Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Bucaramanga', 'Cartagena', 'Cúcuta', 'Ibagué', 'Manizales', 'Pasto', 'Valledupar', 'Santa Marta' 'Montería', 'Villavicencio', 'Neiva', 'Popayán', 'Sincelejo', 'Riohacha', 'Palmira', 'Sahagún', 'Villa del Rosario', 'Tuluá', 'Barrancabermeja', 'Tunja', 'Chía', 'Yopal', 'Maicao', 'Quibdó', 'Tumaco', 'Arauca', 'Cartago', 'Apartadó', 'Jamundí', 'Girardot', 'Fusagasugá', 'Piedecuesta', 'Facatativá', 'Pitalito', 'Ciénaga', 'Ipiales', 'Manaure', 'Zipaquirá', 'Uribia', 'Buga', 'Tierralta', 'Sogamoso Duitama', 'Ocaña', 'Caucasia', 'Aguachica', 'Magangué', 'Montelíbano', 'La Dorada', 'Tiquisio', 'Lorica', 'Santander de Quilichao', 'Rionegro', 'San Vicente del Caguán', 'Sabanalarga', 'San José del Guaviare', 'Espinal', 'Fundación', 'Santa Rosa de Cabal', 'Calarcá' 'Playa Blanca', 'Chigorodó', 'Caldas', 'Puerto Asís', 'Cereté', 'Arjona', 'San Andrés', 'Puerto Boyacá', 'El Carmen de Bolívar', 'Chiquinquirá', 'Planeta Rica', 'Turbaco', 'Morroa', 'Orito', 'La Plata', 'Junain', 'La Hormiga', 'Saravena', 'Pamplona', 'San Marcos']

        df = pd.read_csv(filename)
        if args.country:
            new_df = df.loc[df['location'].isin(city_list)]
            counts = new_df['predicted_emotion'].value_counts()
            total_tweets = len(new_df)
            for emotion in emotions:
                count = counts.get(emotion, 0)
                if normalize:
                    count /= total_tweets
                emotion_counts[emotion][date] = count
        else:
            counts = df['predicted_emotion'].value_counts()
            total_tweets = len(df)
            for emotion in emotions:
                count = counts.get(emotion, 0)
                if normalize:
                    count /= total_tweets
                emotion_counts[emotion][date] = count

    return emotion_counts

# plot the emotion counts in a time series
def plot_emotion_counts(emotion_counts, output_dir, normalize):
    # plot individual graphs
    for emotion, counts in emotion_counts.items():
        dates, counts = zip(*sorted(counts.items())) 
        plt.figure(figsize=(10,6)) 
        plt.plot(dates, counts, marker='o')
        plt.title(emotion.capitalize() + " Time Series")
        plt.xlabel('Date')
        plt.ylabel('Emotion Count')
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        filename = f'{emotion}_time-series'
        if normalize:
            filename += '_normalized'
        plt.savefig(os.path.join(output_dir, filename + '.png'))
        plt.close()
    # plot combined graph
    plt.figure(figsize=(10,6))
    for emotion, counts in emotion_counts.items():
        # if emotion != 'others':
        dates, counts = zip(*sorted(counts.items()))
        plt.plot(dates, counts, marker='o', label=emotion.capitalize())
    if args.country:
        plt.title("All Emotions Time Series " + args.country)
        plt.xlabel('Date')
        plt.ylabel('Emotion Count')
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.legend(loc='upper right')
        filename = 'all_emotions_time-series'
        if normalize:
            filename += '_normalized'
        plt.savefig(os.path.join(output_dir, filename + '.png'))
        plt.close()
    else:
        plt.title("All Emotions Time Series")
        plt.xlabel('Date')
        plt.ylabel('Emotion Count')
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.legend(loc='upper right')
        filename = 'all_emotions_time-series'
        if normalize:
            filename += '_normalized'
        plt.savefig(os.path.join(output_dir, filename + '.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Process and plot emotion counts.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing labeled tweets')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots and csv file')
    parser.add_argument('--normalize', default=False, help='Normalize emotion counts to ratios')
    parser.add_argument('--country', type=str, help='Country to get city list for')
    global args
    args = parser.parse_args()
    # directories for the labeled tweets and output file containing emo counts
    input_dir = args.input_dir
    output_dir = args.output_dir
    # check if output_dir exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = 'emotion_counts'
    if args.normalize:
        output_filename += '_normalized'
    output_file = os.path.join(output_dir, output_filename + '.csv')
    # call necessary functions
    emotion_counts = process_files(input_dir, args.normalize)
    emotion_counts_df = pd.DataFrame(emotion_counts)
    emotion_counts_df.to_csv(output_file)
    plot_emotion_counts(emotion_counts, output_dir, args.normalize)

if __name__ == "__main__":
    main()
