import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

def print_itemsets_table(frequent_itemsets, max_length=4):
    for length in range(1, max_length + 1):
        items = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == length)]
        if not items.empty:
            print(f"\n📏 L({length}) – קבוצות שכיחות בגודל {length}:")
            df_print = items.copy()
            df_print['itemsets'] = df_print['itemsets'].apply(lambda x: ', '.join(sorted(list(x))))
            df_print = df_print.rename(columns={'itemsets': 'Items', 'support': 'Support'})
            print(df_print[['Items', 'Support']].to_string(index=False))

def print_itemsets_as_student_style(frequent_itemsets, total_rows):
    groups = frequent_itemsets.groupby(frequent_itemsets['itemsets'].apply(lambda x: len(x)))
    for k in sorted(groups.groups.keys()):
        group = groups.get_group(k)
        print(f"\nSize of set of large itemsets L({k}): {len(group)}\n")
        print(f"Large Itemsets L({k}):")
        for _, row in group.iterrows():
            items_str = " ".join(sorted(str(i) for i in row['itemsets']))
            count = int(round(row['support'] * total_rows))
            print(f"{items_str}: {count}")

def print_rules_table(rules_df, total_rows, title=" חוקים חזקים"):
    if rules_df.empty:
        print(f"{title}:\n לא נמצאו חוקים חזקים תחת התנאים שנבחרו.")
    else:
        rules_to_show = rules_df.copy()
        rules_to_show['antecedents'] = rules_to_show['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules_to_show['consequents'] = rules_to_show['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules_to_show['count'] = (rules_to_show['support'] * total_rows).round().astype(int)
        rules_to_show = rules_to_show.rename(columns={
            'antecedents': 'Antecedent',
            'consequents': 'Consequent',
            'support': 'Support',
            'confidence': 'Confidence',
            'lift': 'Lift'
        })
        rules_to_show = rules_to_show.sort_values(by='Lift', ascending=False)
        # הוספת אינדקס של שורה
        rules_to_show.reset_index(inplace=True)
        rules_to_show.rename(columns={'index': 'Rule #'}, inplace=True)
        print(f"\n{title} (כמות: {len(rules_to_show)}):")
        pd.options.display.float_format = '{:.3f}'.format
        print(rules_to_show[['Rule #', 'Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift', 'leverage', 'conviction', 'count']].to_string(index=False))

# === שלב ההכנה לפי ממן 21 + ממן 22 ===
print("\n\n===========================")
print(" חלק א: ניתוח כלל האוכלוסייה")
print("===========================\n")

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df[df['gender'] != 'Other']
df.drop(columns=['id'], inplace=True)
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
df['smoking_status'] = df['smoking_status'].fillna("Unknown")
df['stroke'] = df['stroke'].astype(int)

# דיסקרטיזציה חכמה (לפי ממן 22)
df['bmi_group'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, df['bmi'].max()], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Adult', 'Mid-age', 'Old'])
df['glucose_group'] = pd.cut(df['avg_glucose_level'], bins=[0, 90, 140, 200, df['avg_glucose_level'].max()], labels=['Low', 'Normal', 'High', 'Very High'])

# משתנים מורכבים בינאריים לפי ממן 21
df['is_old'] = (df['age'] > 60).astype(int)
df['high_glucose'] = (df['avg_glucose_level'] > 140).astype(int)
df['high_bmi'] = (df['bmi'] > 30).astype(int)
df['married_old'] = ((df['ever_married'] == 'Yes') & (df['is_old'] == 1)).astype(int)
df['urban_and_private'] = ((df['Residence_type'] == 'Urban') & (df['work_type'] == 'Private')).astype(int)

# שלב: מיזוג קבוצות קליניות חלשות לקטגוריות חדשות
df['bmi_group_merged'] = df['bmi_group'].replace({
    'Underweight': 'Low',
    'Normal': 'Normal',
    'Overweight': 'High',
    'Obese': 'High'
})

df['glucose_group_merged'] = df['glucose_group'].replace({
    'Low': 'Low',
    'Normal': 'Normal',
    'High': 'High',
    'Very High': 'High'
})

df['age_group_merged'] = df['age_group'].replace({
    'Young': 'Young',
    'Adult': 'Adult',
    'Mid-age': '60plus',
    'Old': '60plus'
})

df['hypertension_str'] = df['hypertension'].map({0: 'no_hypertension', 1: 'has_hypertension'})
df['heart_disease_str'] = df['heart_disease'].map({0: 'no_heart_disease', 1: 'has_heart_disease'})


# קידוד One-Hot לתכונות קטגוריאליות כולל קבוצות דיסקרטיות
df_encoded = pd.get_dummies(df[[
    'gender', 'ever_married', 'work_type', 'Residence_type',
    'smoking_status', 'bmi_group_merged', 'age_group_merged', 'glucose_group_merged',
'hypertension_str', 'heart_disease_str'
]])


# הוספת משתנים בינאריים חכמים
df_encoded['is_old'] = df['is_old']
df_encoded['high_glucose'] = df['high_glucose']
df_encoded['high_bmi'] = df['high_bmi']
df_encoded['married_old'] = df['married_old']
df_encoded['urban_and_private'] = df['urban_and_private']
df_encoded['stroke'] = df['stroke']

# ביאניזציה
binary_cols = [col for col in df_encoded.columns if df_encoded[col].dropna().isin([0, 1]).all()]
df_apriori = df_encoded[binary_cols].astype(bool)

columns_to_drop = [
   # 'gender_Female', 'gender_Male',
    #'smoking_status_Unknown'
]
df_apriori = df_apriori.drop(columns=[col for col in columns_to_drop if col in df_apriori.columns])


# Apriori
frequent_itemsets = apriori(df_apriori, min_support=0.4, use_colnames=True)
print_itemsets_table(frequent_itemsets)
print_itemsets_as_student_style(frequent_itemsets, df.shape[0])

# חוקים
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# סינון חוקים לא טריוויאליים ולפי מדדים גבוהים
irrelevant_terms = [ 'gender', 'age_group', 'bmi_group_Normal']

def is_informative(items):
    return all(not any(term in str(item) for term in irrelevant_terms) for item in items)

rules = rules[
    (rules['lift'] > 1.2) &
    (rules['confidence'] >= 0.6) &
    (rules['support'] >= 0.4) &
    (rules['leverage'] > 0.03) &
    (rules['conviction'] > 1.1) &
    (rules['antecedents'].apply(lambda x: len(x) >= 2 and is_informative(x))) &
    (rules['consequents'].apply(lambda x: is_informative(x))) &
(rules['antecedents'] != rules['consequents'])  # אין מצב של חוק מסוג X=>X
]


# רשימה של תכונות קליניות בלבד שמותר להופיע בחוקים
clinical_terms = ['ever_married', 'work_type', 'Residence_type', 'hypertension', 'heart_disease', 'high_glucose', 'high_bmi', 'bmi_group', 'glucose_group', 'age_group', 'is_old', 'married_old', 'stroke']

def contains_only_clinical(items):
    return all(any(term in str(item) for term in clinical_terms) for item in items)

# סינון נוסף לפי קליניות בלבד
rules = rules[rules['antecedents'].apply(contains_only_clinical) & rules['consequents'].apply(contains_only_clinical)]

# סינון חוקים שהתוצאה שלהם היא stroke=1 בלבד
stroke_rules = rules[rules['consequents'].apply(lambda x: any('stroke' in s and '_1' in s for s in x))]

# סינון נוסף: רק חוקים עם lift גבוה מ-1.2 (מחזק את הקשר)
stroke_rules = rules[
    rules['consequents'].apply(lambda x: any('stroke' in item for item in x)) &
    (rules['lift'] > 1.2) &
    (rules['confidence'] > 0.7) &
    (rules['support'] >= 0.4)
]

# הסרת חוקים טריוויאליים (שמורכבים רק מ-feature אחד פשוט מדי כמו 'gender_Female')
stroke_rules = stroke_rules[stroke_rules['antecedents'].apply(lambda x: len(x) >= 2)]


# הדפסת החוקים שנשארו
print_rules_table(stroke_rules, df[df['stroke'] == 1].shape[0], " 💥 חוקים חזקים מתוך חולי שבץ בלבד עם lift > 1.2 ו־≥2 תנאים")


# === שלב נוסף – ניתוח רק על חולי שבץ (stroke=1) ===
print("\n\n===========================")
print(" חלק ב: ניתוח על subset של חולי שבץ בלבד")
print("===========================\n")

df_stroke = df_encoded[df_encoded['stroke'] == 1].copy()
df_apriori_stroke = df_stroke[binary_cols].astype(bool)

frequent_itemsets_stroke = apriori(df_apriori_stroke, min_support=0.4, use_colnames=True, max_len=4)
print_itemsets_table(frequent_itemsets_stroke)
print_itemsets_as_student_style(frequent_itemsets_stroke, df_stroke.shape[0])

rules_stroke = association_rules(frequent_itemsets_stroke, metric="confidence", min_threshold=0.6)
print_rules_table(rules_stroke, df_stroke.shape[0], " חוקים כלליים מתוך חולי שבץ בלבד")
strong_rules_stroke = rules_stroke[rules_stroke['lift'] > 1.0]
print_rules_table(strong_rules_stroke, df_stroke.shape[0], " חוקים חזקים מתוך חולי שבץ בלבד עם lift > 1")

from tabulate import tabulate

# הכנת הדאטה לפרינט
strong_rules_stroke_to_save = strong_rules_stroke.copy()
strong_rules_stroke_to_save['Antecedent'] = strong_rules_stroke_to_save['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
strong_rules_stroke_to_save['Consequent'] = strong_rules_stroke_to_save['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
strong_rules_stroke_to_save['count'] = (strong_rules_stroke_to_save['support'] * df_stroke.shape[0]).round().astype(int)

# בחירת העמודות בפורמט הרצוי
cols = ['Antecedent', 'Consequent', 'support', 'confidence', 'lift', 'leverage', 'conviction', 'count']
df_to_export = strong_rules_stroke_to_save[cols].copy()
df_to_export.insert(0, "Rule #", range(len(df_to_export)))  # מוסיף עמודת Rule #

# שמירה לקובץ txt עם טבלאה מיושרת
with open("strong_stroke_rules_tabulated.txt", "w", encoding="utf-8") as f:
    f.write("💥 חוקים חזקים מתוך חולי שבץ בלבד עם lift > 1\n\n")
    f.write(tabulate(df_to_export, headers="keys", tablefmt="grid", showindex=False))
