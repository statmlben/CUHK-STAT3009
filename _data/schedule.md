> "*All models are wrong, but some are useful." — George E. P. Box*

## <span style="color:#A04000"> 🗓️ Schedule (tentative) </span>

<table class="table">
  <colgroup>
    <col style="width:10%">
    <col style="width:20%">
    <col style="width:40%">
    <col style="width:10%">
    <col style="width:10%">
  </colgroup>
  <thead>
  <tr class="active">
    <th>Date</th>
    <th>Description</th>
    <th>Course Materials</th>
    <th>Events</th>
    <th>Deadlines</th>
  </tr>
  </thead>
  <tr>
    <td>Prepare</td>
    <td>Course information
      <br>
      [<a href="../_pages/STAT3009/Lec-pre/S.pdf">slides</a>]
      <br><br>
      Python Tutorial
      <br>
      [<a href="https://youtu.be/rfscVS0vtbw">YouTube</a>]
      <br><br>
      Numpy, Pandas, Matplotlib
      <br>
      [<a href="https://cs231n.github.io/python-numpy-tutorial/">notes</a>]
      [<a href="https://youtu.be/LHBE6Q9XlzI">YouTube</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="https://www.learnpython.org/">learnpython.org</a></li>
        <li><a href="https://docs.python.org/3/tutorial/">The Python Tutorial</a> (official Python documentation)</li>
      </ol>
    </td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td>Sep 07</td>
    <td>Background and baseline methods
      <br>
      [<a href="../_pages/STAT3009/Lec-baseline/S.pdf">slides</a>]
      [<a href="https://drive.google.com/file/d/1-ARE7b8afzKI6PcC2rt7S8BBmvlG2WOn/view?usp=sharing">colab</a>]
      [<a href="https://github.com/statmlben/CUHK-STAT3009/blob/main/nb_background.ipynb">github</a>]
    </td>
    <td>
      Suggested Readings:

      <ol>
        <li><a href="https://en.wikipedia.org/wiki/Netflix_Prize">Wiki: Netflix Prize</a></li>
        <li><a href="https://cseweb.ucsd.edu/~jmcauley/datasets.html">Recommender Systems Datasets - UCSD CSE</a></li>
      </ol>

​    </td>
​    <td></td>
​    <td></td>
  </tr>

  <tr>
    <td>Sep 14</td>
    <td>Correlation-based RS
      <br>
      [<a href="../_pages/STAT3009/Lec-corr/S.pdf">slides</a>]
      [<a href="https://colab.research.google.com/drive/1C1ldh2FgT1kRz_3wc2OgozQm5GaQGw5t?usp=sharing">colab</a>]
      [<a href="https://github.com/statmlben/CUHK-STAT3009/blob/main/nb_correlation.ipynb">github</a>]
    </td>
    <td>
      Suggested Readings:

      <ol>
        <li><a href="https://www.ibm.com/hk-en/topics/knn">K-Nearest Neighbors Algorithm</a></li>
        <li><a href="https://www.machinelearningplus.com/nlp/cosine-similarity/">Cosine Similarity in NLP</a></li>
        <li><a href="https://stats.stackexchange.com/a/451376/">Curse of high dimensionality</a></li>
      </ol>

​    </td>
​    <td></td>
​    <td></td>
  </tr>


  <tr class="warning">
    <td>Sep 21</td>
    <td>⏰ Quiz 1: implement baseline methods and correlation-based RS
      <br>
      [<a href="../_pages/STAT3009/quiz/quiz1/instruct/instruct.pdf">instruct</a>]
      [<a href="../_pages/STAT3009/quiz/quiz1/sum/report.pdf">report</a>]
    </td>
    <td>
      <i class="fa fa-clock-o"></i> InClass quiz<br>via Kaggle (link on BlackBoard)</td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td>Sep 28</td>
    <td>ML overview
      <br>
      [<a href="../_pages/STAT3009/Lec-ML/S.pdf">slides</a>]
      [<a href="https://colab.research.google.com/drive/1hrpclpigZgRGFoAbgZ5V6Sk-iLS8QiYQ?usp=sharing">colab</a>]
      [<a href="https://github.com/statmlben/CUHK-STAT3009/blob/main/nb_ml.ipynb">github</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li>Chapters 2-3 in <a href="https://hastie.su.domains/Papers/ESLII.pdf">The Elements of Statistical Learning</a></li>
        <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">Linear regression in sklearn</a></li>
      </ol>
    </td>
    <td>
    HW 1 <b><font color="#c0842b">release</font></b>
    <br>
    [<a href="https://colab.research.google.com/drive/1CKvwWySUnvHVoSUmpzWXiQTdMnxLYgg8?usp=sharing">colab</a>]
    </td>
    <td></td>
  </tr>

<tr>
    <td>Oct 05</td>
    <td>Matrix factorization I: ALS/BCD
      <br>
      [<a href="../_pages/STAT3009/Lec-MF/S.pdf">slides</a>]
      [<a href="https://colab.research.google.com/drive/1PJ8lTWvS2xPA3Cske38fqvsrruxR2gCU?usp=sharing">colab</a>]
      [<a href="https://github.com/statmlben/CUHK-STAT3009/blob/main/nb_mf.ipynb">github</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="https://sifter.org/simon/journal/20061211.html">Netflix Update: Try This at Home</a> (first one applied MF in RS)</li>
        <li><a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5197422">Matrix factorization techniques for recommender systems</a></li>
        <li><a href="https://www.benfrederickson.com/matrix-factorization/">Finding Similar Music using Matrix Factorization</a></li>
        <li><a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5197422">Matrix factorization techniques for recommender systems</a></li>
        <li><a href="https://jmlr.org/papers/volume16/hastie15a/hastie15a.pdf">Matrix completion and low-Rank SVD via fast alternating least squares</a></li>
        <li><a href="https://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/22-coord-desc.pdf">Coordinate Descent (Slides by Ryan Tibshirani)</a></li>
     </ol>
    </td>
    <td>
    HW 2 <b><font color="#c0842b">release</font></b>
    <br>
    [<a href="https://colab.research.google.com/drive/1SSsHWlJsia1CmnYl_PBMVBsvQbV_fnzf?usp=sharing">colab</a>]
    </td>
    <td>
    HW 1 <b><font color="#673ab7">due</font></b>
    <br>
    [<a href="https://colab.research.google.com/drive/1N_TkhFG1q2TF96uz_ul9LTuijwnvGr5s?usp=sharing">sol</a>]
    </td>
</tr>

<tr>
    <td>Oct 12</td>
    <td>Matrix factorization II: SGD
      <br>
      [<a href="../_pages/STAT3009/Lec-SGD/S.pdf">slides</a>]
      [<a href="https://colab.research.google.com/drive/19psY2pGjdjO2y0O1K9Eu28KdlfeuzVTO?usp=sharing">colab</a>]
      [<a href="https://github.com/statmlben/CUHK-STAT3009/blob/main/nb_SGD.ipynb">github</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="https://scikit-learn.org/stable/modules/sgd.html">Stochastic Gradient Descent </a>(sklearn documentation)</li>
        <li><a href="https://realpython.com/gradient-descent-algorithm-python/">Stochastic Gradient Descent Algorithm With Python and NumPy </a></li>
      </ol></td>
    <td>
    Proj 1 <b><font color="#c0842b">release</font></b>
    <br>
    [<a href="../_pages/STAT3009/proj1/instruct/instruct.pdf">instruct</a>]
    </td>
    <td>
    HW 2 <b><font color="#673ab7">due</font></b>
    <br>
    [<a href="https://colab.research.google.com/drive/16um3OeVXPB4sDhwfJbCI3S6tbBVqdVUC?usp=sharing">sol</a>]
    </td>
</tr>


<tr>
    <td>Oct 19</td>
    <td>Factorization Meets the Neighborhood
      <br>
      [<a href="../_pages/STAT3009/Lec-MF+KNN/S.pdf">slides</a>]
      [<a href="https://github.com/statmlben/CUHK-STAT3009/blob/main/nb_mfpp.ipynb">github</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf">Improving regularized singular value decomposition for collaborative filtering </a></li>
        <li><a href="https://dl.acm.org/doi/abs/10.1145/1401890.1401944">Factorization meets the neighborhood: a multifaceted collaborative filtering model </a></li>
        <li><a href="https://www.jmlr.org/papers/v20/17-629.html">Smooth neighborhood recommender systems </a></li>
      </ol></td>
    <td></td>
    <td></td>
</tr>

<tr>
    <td>Oct 26</td>
    <td>Case Study: MovieLens
      <br>
      [<a href="../_pages/STAT3009/Lec-EDA/S.pdf">slides</a>]
      [<a href="https://github.com/statmlben/CUHK-STAT3009/blob/main/nb_EDA.ipynb">github</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="https://www.kaggle.com/competitions/home-depot-product-search-relevance">Home Depot Product Search Relevance </a>(Kaggle competition)</li></ol></td>
    <td>
    </td>
    <td>
    Proj 1 <b><font color="#673ab7">due</font></b>
    </td>
</tr>

<tr>
    <td>Nov 02</td>
    <td>Neural Networks
      <br>
      [<a href="../_pages/STAT3009/Lec-NN/S.pdf">slides</a>]
      [<a href="https://github.com/statmlben/CUHK-STAT3009/blob/main/nb_nn.ipynb">github</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li>Chapter 11 in <a href="https://hastie.su.domains/Papers/ESLII.pdf">The Elements of Statistical Learning</a></li>
        <li><a href="http://neuralnetworksanddeeplearning.com/index.html">Neural Networks and Deep Learning </a>(free online book)</li>
      </ol>
    </td>
    <td>
    Proj 2 <b><font color="#c0842b">release</font></b>
    <br>
    [<a href="../_pages/STAT3009/proj2/instruct/instruct.pdf">instruct</a>]
    </td>
    <td></td>
    <td></td>
</tr>

<tr>
    <td>Nov 09</td>
    <td>Neural collaborative filtering
      <br>
      [<a href="../_pages/STAT3009/Lec-ncf/S.pdf">slides</a>]
      [<a href="https://github.com/statmlben/CUHK-STAT3009/blob/main/nb_ncf.ipynb">github</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="https://arxiv.org/abs/1708.05031">Neural Collaborative Filtering </a>(original paper of NCF)</li>
        <li><a href="https://arxiv.org/pdf/1707.07435.pdf">Deep Learning based Recommender System: A Survey and New Perspectives </a></li>
        <li><a href="https://www.tensorflow.org/recommenders">TensorFlow Recommenders</a></li>
        <li><a href="https://medium.com/analytics-vidhya/understanding-embedding-layer-in-keras-bbe3ff1327ce">Understanding Embedding Layer in Keras (NLP) </a></li>
      </ol>
    </td>
    <td>
    HW 3 <b><font color="#c0842b">release</font></b>
    <br>
    [<a href="https://colab.research.google.com/drive/12ImcXQ8KY4G6hAa5Kq9UyLiWAZ2nVQc7?usp=sharing">colab</a>]
    </td>
    <td></td>
</tr>

<tr>
    <td>Nov 16</td>
    <td>Side information
      <br>
      [<a href="../_pages/STAT3009/Lec-side/S.pdf">slides</a>]
      [<a href="https://github.com/statmlben/CUHK-STAT3009/blob/main/nb_side.ipynb">github</a>]
    </td>
    <td>
      Suggested Readings:
      <ol>
        <li><a href="https://www.kaggle.com/competitions/home-depot-product-search-relevance">Home Depot Product Search Relevance </a>(Kaggle competition)</li>
        <li><a href="https://blog.tensorflow.org/2020/09/introducing-tensorflow-recommenders.html"> Introducing TensorFlow Recommenders </a></li>
        <li><a href="https://research.netflix.com/publication/%20Deep%20Learning%20for%20Recommender%20Systems%3A%20A%20Netflix%20Case%20Study"> Deep learning for recommender systems: A Netflix case study </a></li>
      </ol>
    </td>
    <td></td>
    <td>
    HW 3 <b><font color="#673ab7">due</font></b>
    </td>
</tr>

<tr>
    <td><del>Nov 23</del></td>
    <td><del>Model Averaging</del>
      <br>
      <!-- [<a href="https://www.dropbox.com/s/2iptwnmhw3v459x/beamerthemeNord.pdf?dl=0">slides</a>] -->
      <!-- [<a href="https://www.dropbox.com/s/g28znbig02f6oiq/beamerthemeNord.pdf?dl=0">colab</a>]
      [<a href="https://github.com/statmlben/CUHK-STAT3009/blob/main/notebook1.ipynb">github</a>] -->
    </td>
    <td>
    Suggested Readings:
      <ol>
        <li>Chapters 8 and 16 in <a href="https://hastie.su.domains/Papers/ESLII.pdf">The Elements of Statistical Learning</a></li>
        <li><a href="https://builtin.com/machine-learning/ensemble-model">Ensemble Models: What Are They and When Should You Use Them?</a></li>
        <li><a href="https://github.com/yzhao062/combo">combo: A Python Toolbox for Machine Learning Model Combination</a></li>
      </ol>
    </td>
    <td></td>
    <td></td>
</tr>

  <tr class="warning">
    <td>Nov 30</td>
    <td> ⏰ Quiz 2: Math & Python
      <br>
      <!-- [<a href="readings/python_tutorial.ipynb">kaggle competition</a>] -->
    </td>
    <td>
      <i class="fa fa-clock-o"></i> InClass quiz<br>
    </td>
    <td></td>
    <td></td>
  </tr>

  <tr class="warning">
    <td>-</td>
    <td></td>
    <td></td>
    <td></td>
    <td>
      Proj 2 <b><font color="red">due</font></b>
      <!-- [<a href="project/project-report-instructions-2022.pdf">instructions</a>] -->
    </td>
  </tr>
<html>