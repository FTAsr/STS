BoW model requires word2vec pre-trained Google News corpus (3 billion running words) word vector model (3 million 300-dimension English word vectors).

It can be downloaded from the following URL:

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

<h1> SVR model performance on combined feature vector(len 625) </h1> <br />

************ SUMMARY DEV***********<br />
Train data size: 4500 <br />
Dev data size: 500 <br />
Dev Pearson: 0.683097988355 <br />
Dev Spearman: 0.610643840413 <br />
Dev MSE: 0.540307353374 <br />
********************************<br />

************ SUMMARY TEST***********<br />
Test data size: 4927<br />
Test Pearson: 0.690650791745<br />
Test Spearman: 0.608926069542<br />
Test MSE: 0.53565299858<br />
********************************
<h1>LR model performance on combined feature vector(len 625) </h1> <br />

************ SUMMARY DEV***********<br />
Train data size: 4500 <br />
Dev data size: 500 <br />
Dev Pearson: 0.817104720634 <br />
Dev Spearman: 0.760527966441 <br />
Dev MSE: 0.339688015953 <br />
********************************

************ SUMMARY TEST***********<br />
Test data size: 4927 <br />
Test Pearson: 0.82175847506 <br />
Test Spearman: 0.741193508775 <br />
Test MSE: 0.334077946145 <br />
********************************

