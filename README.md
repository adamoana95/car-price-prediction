# car-price-prediction
task final


Car Price Prediction

1. Descrierea proiectului

Acest proiect are ca scop dezvoltarea unui model de Machine Learning pentru estimarea prețului mașinilor second-hand.

Modelul primește informații despre o mașină, precum marca, modelul, anul fabricației, kilometrajul, tipul de combustibil, volumul motorului, culoarea, transmisia, tracțiunea și segmentul mașinii și estimează prețul acesteia în USD.

Proiectul este realizat ca o problemă de regresie, deoarece variabila țintă, priceusd, este o valoare numerică continuă.

⸻

2. Setul de date

Proiectul utilizează setul de date cars.csv, care conține informații despre mașini second-hand.

Principalele coloane din setul de date sunt:

* make – marca mașinii;
* model – modelul mașinii;
* priceUSD – prețul mașinii în USD, variabila țintă;
* year – anul fabricației;
* condition – starea mașinii;
* mileage(kilometers) – kilometrajul în kilometri;
* fuel_type – tipul de combustibil;
* volume(cm3) – volumul motorului;
* color – culoarea;
* transmission – tipul transmisiei;
* drive_unit – tipul tracțiunii;
* segment – segmentul mașinii.

Setul de date conține atât variabile numerice, cât și variabile categorice.

⸻

3. Curățarea datelor

În etapa de curățare au fost realizate următoarele operații:

* standardizarea numelor coloanelor;
* eliminarea spațiilor inutile din valorile text;
* transformarea valorilor categorice într-un format consistent;
* identificarea valorilor lipsă și a valorilor care reprezintă date necunoscute;
* conversia coloanelor numerice la tipurile corespunzătoare;
* identificarea valorilor invalide pentru preț, kilometraj, an și volum motor;
* tratarea valorilor extreme ale kilometrajului.

Pentru kilometraj, valorile mai mari de 500.000 km au fost considerate nerealiste și au fost transformate în valori lipsă, fără eliminarea întregului rând.

Valorile lipsă sunt tratate ulterior în etapa de preprocesare folosind imputarea statistică.

⸻

4. Feature Engineering

Pentru a obține informații suplimentare utile pentru modelele de Machine Learning, au fost create următoarele caracteristici:

car_age

Vârsta mașinii a fost calculată folosind anul de referință 2026:

car_age = 2026 - year

mileage_per_year

A fost calculat kilometrajul mediu anual:

mileage_per_year = mileage_km / car_age

engine_volume_liters

Volumul motorului exprimat în cm³ a fost transformat în litri:

engine_volume_liters = volume_cm3 / 1000

is_newer_car

A fost creată o caracteristică categorică pentru identificarea mașinilor fabricate începând cu anul 2010.

is_high_mileage

A fost creată o caracteristică pentru identificarea mașinilor cu un kilometraj de cel puțin 300.000 km.

brand_model

Marca și modelul au fost combinate într-o singură caracteristică:

brand_model = make + model

Caracteristicile originale au fost păstrate pentru a putea fi utilizate împreună cu cele noi.

⸻

5. Preprocesarea datelor

Datele au fost împărțite în două categorii:

Caracteristici numerice

* year
* mileage_km
* volume_cm3
* car_age
* mileage_per_year
* engine_volume_liters

Pentru valorile lipsă a fost utilizată imputarea cu mediana, iar apoi valorile au fost standardizate folosind StandardScaler.

Caracteristici categorice

* make
* model
* condition
* fuel_type
* color
* transmission
* drive_unit
* segment
* is_newer_car
* is_high_mileage
* brand_model

Valorile lipsă au fost completate folosind cea mai frecventă valoare, iar variabilele categorice au fost transformate folosind OneHotEncoder.

Pentru valorile categorice necunoscute în timpul predicției a fost utilizată opțiunea handle_unknown="ignore".

Preprocesarea și modelul sunt integrate într-un Pipeline, pentru ca aceleași transformări să fie aplicate atât datelor de antrenare, cât și datelor de test.

⸻

6. Modelele testate

Pentru problema de regresie au fost testate următoarele modele:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

Toate modelele au fost evaluate folosind același set de antrenare și testare și aceeași metodă de preprocesare.

Datele au fost împărțite astfel:

* 80% – date de antrenare;
* 20% – date de testare.

Pentru reproducibilitatea rezultatelor a fost utilizat random_state=42.

⸻

7. Compararea modelelor

Rezultatele obținute sunt:

Model	MAE	MSE	RMSE	R²
Linear Regression	1989.01	17,584,860.00	4193.43	0.7318
Decision Tree	1361.39	11,168,470.00	3341.93	0.8297
Random Forest	1058.74	6,927,214.63	2631.96	0.8943
Gradient Boosting	1553.25	9,788,055.00	3128.59	0.8507

Interpretarea rezultatelor

MAE reprezintă eroarea absolută medie dintre prețul real și cel prezis. O valoare mai mică indică predicții mai precise.

RMSE penalizează mai puternic erorile mari. Și în cazul acestei metrici, valoarea mai mică este mai bună.

R² indică proporția din variația prețului care poate fi explicată de model. O valoare mai apropiată de 1 indică o performanță mai bună.

Din rezultatele obținute se observă că Random Forest Regressor are cele mai bune rezultate pentru toate cele trei criterii principale: MAE, RMSE și R².

⸻

8. Alegerea modelului final

Modelul ales pentru versiunea finală a proiectului este Random Forest Regressor.

Acesta a obținut:

* MAE = 1058.74 USD
* RMSE = 2631.96 USD
* R² = 0.8943

Random Forest a fost ales deoarece a obținut cea mai mică eroare medie absolută, cea mai mică eroare RMSE și cel mai mare coeficient R² dintre modelele testate.

Modelul final este salvat în:

models/random_forest_model.joblib

Modelul salvat include și etapa de preprocesare, astfel încât poate fi utilizat ulterior pentru predicții fără a reconstrui manual transformările aplicate datelor.

⸻

9. Cum se rulează proiectul

9.1. Instalarea proiectului

Se clonează repository-ul:

git clone https://github.com/adamoana95/car-price-prediction.git
cd car-price-prediction

9.2. Instalarea bibliotecilor

Se recomandă crearea unui mediu virtual:

python -m venv .venv

Activarea acestuia pe Windows:

.venv\Scripts\activate

Instalarea dependențelor:

pip install -r requirements.txt

9.3. Rularea etapelor proiectului

Scripturile pot fi rulate din directorul src în următoarea ordine:

python data_cleaning.py
python feature_engineering.py
python model_training.py
python model_evaluation.py
python model_comparison.py
python final_model.py

Preprocesarea este utilizată de pipeline-urile modelelor și nu trebuie rulată separat pentru a antrena modelul final.

9.4. Notebook-ul

Notebook-ul Jupyter din directorul notebooks poate fi deschis cu:

jupyter notebook

sau:

jupyter lab

Notebook-ul poate fi utilizat pentru explorarea și prezentarea etapelor proiectului.

⸻

10. Structura proiectului

car-price-prediction/
│
├── data/
│   ├── cars.csv
│   ├── cars_cleaned.csv
│   └── cars_features.csv
│
├── models/
│   └── random_forest_model.joblib
    └── linear_regression_model.joblib
│
├── notebooks/
│   └── car_price_prediction.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── model_comparison.py
│   └── final_model.py
│
├── README.md
├── requirements.txt
└── .gitattributes

⸻

11. Concluzie

Proiectul demonstrează utilizarea unui flux complet de Machine Learning pentru o problemă de regresie: curățarea datelor, ingineria caracteristicilor, preprocesarea variabilelor, antrenarea mai multor modele, evaluarea și compararea acestora.

Dintre modelele analizate, Random Forest Regressor a obținut cele mai bune rezultate, cu un R² de 0.8943 și o eroare absolută medie de aproximativ 1059 USD. Din acest motiv, acesta a fost ales ca model final al proiectului.
