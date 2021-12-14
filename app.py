from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import os
from utils import make_prediction_kingdom, make_prediction_dnatype
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename



columns = ['Ncodons', 'UUU', 'UUC', 'UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG',
       'AUU', 'AUC', 'AUA', 'AUG', 'GUU', 'GUC', 'GUA', 'GUG', 'GCU', 'GCC',
       'GCA', 'GCG', 'CCU', 'CCC', 'CCA', 'CCG', 'UGG', 'GGU', 'GGC', 'GGA',
       'GGG', 'UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC', 'ACU', 'ACC', 'ACA',
       'ACG', 'UAU', 'UAC', 'CAA', 'CAG', 'AAU', 'AAC', 'UGU', 'UGC', 'CAU',
       'CAC', 'AAA', 'AAG', 'CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG', 'GAU',
       'GAC', 'GAA', 'GAG', 'UAA', 'UAG', 'UGA']



kingdom_classes = ['Archea', 'Bacteria', 'Eukaryots']
DNAtype_classes = ['genomic', 'mitochondrial', 'chloroplast']




app=Flask(__name__,template_folder='templetes', static_folder = 'img')

app.config['UPLOAD_FOLDER'] = 'img'




@app.route("/")
def index():
    return render_template('index.html')



@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        file = request.files['upload']
        if file.filename=="":
            return render_template('index.html', text = 'Please choose a file to upload')
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename),"r") as f:
            con = f.readlines()
        text = ''.join(con)


        input_val = list(map(float, text.split(',')))
        data = pd.Series(input_val, columns)
        pred_kingdom, d = make_prediction_kingdom(data, 'kingdom/clf_kingdom_svm.pkl' , 'kingdom/class_encoding_kingdom.pkl', 'kingdom/std_kingdom.pkl', 'kingdom/good_features_kingdom.pkl')
        pred_dnatype = make_prediction_dnatype(data, clf_path = 'DNAtype/clf_best_dnatype.pkl' )
        prediction_text = f'The following organism belong to {kingdom_classes[pred_kingdom[0]]} Kingdom and its DNAtype is {DNAtype_classes[pred_dnatype[0]]}'
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('index.html', prediction_text = prediction_text)


    # 'Ncodons', 'UUU', 'UUC', 'UUA', 'UUG', 'CUU',
    #    'CUC', 'CUA', 'CUG', 'AUU', 'AUC', 'AUA', 'AUG', 'GUU', 'GUC', 'GUA',
    #    'GUG', 'GCU', 'GCC', 'GCA', 'GCG', 'CCU', 'CCC', 'CCA', 'CCG', 'UGG',
    #    'GGU', 'GGC', 'GGA', 'GGG', 'UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC',
    #    'ACU', 'ACC', 'ACA', 'ACG', 'UAU', 'UAC', 'CAA', 'CAG', 'AAU', 'AAC',
    #    'UGU', 'UGC', 'CAU', 'CAC', 'AAA', 'AAG', 'CGU', 'CGC', 'CGA', 'CGG',
    #    'AGA', 'AGG', 'GAU', 'GAC', 'GAA', 'GAG', 'UAA', 'UAG', 'UGA'


if __name__ == '__main__':
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, host='0.0.0.0', port = port)
    # app.run(debug=True)
