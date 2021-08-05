# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 12:32:02 2021
@author: Arunabha
"""

{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "db826834",
   "metadata": {},
   "source": [
    "# The Sparks Foundation \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0e718d73",
   "metadata": {},
   "source": [
    "# Data Science and Business Analytics Internship Task 1 "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2debab73",
   "metadata": {},
   "source": [
    "# Task by: Ashish Handa"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "24a051c4",
   "metadata": {},
   "source": [
    "##  Prediction using Supervised ML\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "feff6ce8",
   "metadata": {},
   "source": [
    "### Step 1: Understand the Problem\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "25792efd",
   "metadata": {},
   "source": [
    "Predict the percentage of a student based on the no. of study hours."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0c0b2d3e",
   "metadata": {},
   "source": [
    "### Step 2: Find the source of data and load it.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cefc49be",
   "metadata": {},
   "source": [
    "Dataset is available at: https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "75cd0150",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b860f98f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours  Scores\n",
       "0    2.5      21\n",
       "1    5.1      47\n",
       "2    3.2      27\n",
       "3    8.5      75\n",
       "4    3.5      30"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"C:\\\\Users\\\\Rahul Handa\\\\Desktop\\Data.csv\")\n",
    "\n",
    "\n",
    "\n",
    "# Printing the first 5 rows of the dataset\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23a4d492",
   "metadata": {},
   "source": [
    "### Step 3: Explore the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9aa89bf3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 25 entries, 0 to 24\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   Hours   25 non-null     float64\n",
      " 1   Scores  25 non-null     int64  \n",
      "dtypes: float64(1), int64(1)\n",
      "memory usage: 464.0 bytes\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "cfe7235b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>25.000000</td>\n",
       "      <td>25.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.012000</td>\n",
       "      <td>51.480000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.525094</td>\n",
       "      <td>25.286887</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.100000</td>\n",
       "      <td>17.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.700000</td>\n",
       "      <td>30.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>4.800000</td>\n",
       "      <td>47.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7.400000</td>\n",
       "      <td>75.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>9.200000</td>\n",
       "      <td>95.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Hours     Scores\n",
       "count  25.000000  25.000000\n",
       "mean    5.012000  51.480000\n",
       "std     2.525094  25.286887\n",
       "min     1.100000  17.000000\n",
       "25%     2.700000  30.000000\n",
       "50%     4.800000  47.000000\n",
       "75%     7.400000  75.000000\n",
       "max     9.200000  95.000000"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6dad4f4c",
   "metadata": {},
   "source": [
    "### Step 4: Visualize the data.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "dade0a47",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "6b987766",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(df.Hours, df.Scores)\n",
    "plt.title(\"Hours Vs Scores\")\n",
    "plt.xlabel(\"------ Hours ----->\")\n",
    "plt.ylabel(\"------ Scores ----->\")\n",
    "plt.grid()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d2a4948e",
   "metadata": {},
   "source": [
    "### Step 5: Model Selection and model training.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "74672dd3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Taking input column in x\n",
    "x = df['Hours']\n",
    "\n",
    "# Taking output/target column in y\n",
    "y = df['Scores']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c0897540",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    2.5\n",
       "1    5.1\n",
       "2    3.2\n",
       "3    8.5\n",
       "4    3.5\n",
       "Name: Hours, dtype: float64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "d97837ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    21\n",
       "1    47\n",
       "2    27\n",
       "3    75\n",
       "4    30\n",
       "Name: Scores, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "8d7ce27b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting the data into train and test for model building\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "20b88c73",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(18, 7)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(x_train), len(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "dc07c3d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "\n",
    "model = LinearRegression()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b0d78c5f",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = x_train.values.reshape(-1, 1)\n",
    "y_train = y_train.values.reshape(-1, 1)\n",
    "\n",
    "x_test = x_test.values.reshape(-1, 1)\n",
    "y_test = y_test.values.reshape(-1, 1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "f77570d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "794102c4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": 
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plotting the mx + c line over test data\n",
    "\n",
    "m = model.coef_[0]\n",
    "c = model.intercept_\n",
    "\n",
    "\n",
    "x_line = np.arange(0, 10, 0.1)\n",
    "\n",
    "y_line = m * x_line + c\n",
    "\n",
    "plt.plot(x_line, y_line, \"r\")\n",
    "\n",
    "plt.scatter(x_test, y_test)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2a9379b6",
   "metadata": {},
   "source": [
    "### Step 6: Model Testing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "e50e20dc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9367661043365055"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.score(x_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b606070c",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(x_test)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "77cbd66f",
   "metadata": {},
   "source": [
    "### What will be predicted score if a student studies for 9.25 hrs/ day?\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "58f59532",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "93.89272889341655"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "user_input = np.array(9.25).reshape(1, -1)\n",
    "\n",
    "\n",
    "answer = model.predict(user_input)\n",
    "\n",
    "answer[0][0]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2e462d13",
   "metadata": {},
   "source": [
    "### Predicted score if a student studies for 9.25 hrs/ day: 93.89272889341655\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
