# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 20:20:53 2025

@author: User
"""

import openai
import numpy as np
# import re
from plyer import notification
import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle
from datetime import datetime
from havruta_HM_thesis_prompts_3_classes import load_trait
from havruta_HM_thesis_load_rules_3_classes import load_rules_with_gpt
from havruta_HM_thesis_classification import classifier_using_rules
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

openai.api_key = os.getenv('OPENAI_API_KEY')
pd.set_option('display.max_colwidth', None)  # Show full text in cells

# import matplotlib.pyplot as plt
# import seaborn as sns

# -----------------------------------------------------------------------

# Load dataset (social media posts + true labels)
def load_posts_and_trait_true_label(personality_trait: str):
    df = pd.read_excel("C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/true_labels_3_classes.xlsx")
    # for _, row in df.iterrows():
    #       row['post1'], row['post2'] = row['post1'].replace("â€™","'"), row['post2'].replace("â€™","'")
    return df[['p', 'post1', 'post2', f'{personality_trait}']]


def custom_fixed_slices(df):
    n_samples = len(df)
    train_size = int(n_samples/2)
    test_size = int(n_samples/2)
    assert train_size + test_size == n_samples
    assert n_samples % test_size == 0

    splits = []
    n_folds = n_samples // test_size

    for i in range(n_folds):
        test_start = i * test_size  
        test_end = test_start + test_size 
        test_data = df.iloc[test_start : test_end] # slice rows correctly

        remaining_indices = list(range(0, test_start)) + list(range(test_end, n_samples))
        train_data  = df.iloc[remaining_indices]  # use .iloc here!

        # val_data = remaining_data.iloc[:val_size]
        # train_data = remaining_data.iloc[val_size:val_size + train_size]

        splits.append({
            "fold": i + 1,
            "train": train_data,
            # "val": val_data,
            "test": test_data
        })

    return splits

def time_stamp(text: str):
        now = datetime.now()
        now_txt = now.strftime("%Y-%m-%d_%H-%M-%S")
        print(text, now_txt)
        return now_txt
  
def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", save_path=None):
    """
    Plots and optionally saves a confusion matrix with white-colored axis ticks.

    Parameters:
    - y_true: list of true class labels
    - y_pred: list of predicted class labels
    - labels: list of class labels (e.g., ["Low", "Moderate", "High"])
    - title: plot title
    - save_path: file path to save the image (e.g., "cmatrix.png"). If None, does not save.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted", color='white')
    plt.ylabel("True", color='white')
    plt.title(title, color='white')
    plt.xticks(color='red')
    plt.yticks(color='red')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, facecolor='black', dpi=300)
    
    plt.show()

def main(personality_trait: str, num_reps: int):
    participants = load_posts_and_trait_true_label(personality_trait)
    dict_class_to_code = {'Unknown': -10, 'Low': 1, 'Moderate': 2, 'High': 3}
    
    # Specify the directory name
    start_txt = time_stamp('START')
    results_dir = 'C:/Users/User/OneDrive/Desktop/Masters/thesis/Results/'
    results_trait_file_name= f'{load_trait(personality_trait)}/'
    timestamp_folder = f'run_{start_txt}/'
    os.mkdir(''.join(results_dir + results_trait_file_name + timestamp_folder))
    path = results_dir + results_trait_file_name + timestamp_folder
    directory_names = ["loaded_rules", "test_set_results"] 
    for directory_name in directory_names:
        # Create the directory
        try:
            os.mkdir(''.join(path + directory_name))
            # print(f"Directory '{directory_name}' created successfully.")
        except FileExistsError: 
            pass
            # print(f"Directory '{directory_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
    all_reps = []
    all_reps_accuracies_rules, all_reps_maes_rules = [], []
    all_reps_accuracies_few, all_reps_maes_few = [], []
    all_reps_accuracies_zero, all_reps_maes_zero = [], []
    
    for rep in range(1, num_reps + 1):
        print('----------Replication', rep, 'Beginess----------')

        rules = []
        time_stamp('Start Rep')
        rep_results_test = []
        
        # Shuffle and reset index
        participants = shuffle(participants).reset_index(drop=True)
        
        splits = custom_fixed_slices(participants)
  
        for fold in splits:
            print(f"      -----Fold {fold['fold']}   (rep {rep}) ------")
            train = fold['train']
            # val = fold['val']
            test = fold['test']
            
            # print('train\n', train[['p']], '\n\n', 'val\n', val['p'], '\n\n' ,'test\n', test['p'])
            train_participants = train['p'].to_list()

            # n = len(train_participants)
            # chunk_size = n // 3  # Integer division
            
            # Create the three sub-lists using slicing
            # list1 = train_participants[0:chunk_size]
            # list2 = train_participants[chunk_size:2 * chunk_size]
            # list3 = train_participants[2 * chunk_size:] # The rest of the list
            

            print('Loading Rules')
            try:
                rules = load_rules_with_gpt(personality_trait, train_participants)
            except:
                print('problem with loading rules')
                print(rules)
                return None
            
            # Concatenate folds and save per rep        
            with open(f"{path}{directory_names[0]}/Loaded rules, Rules rep {rep} fold {fold['fold']}.txt", 'w') as f:
                for item in rules:
                    f.write(str(item) + '\n')    
            print('loading extra_prompt_few_shot')
            # Build few-shot prompt using training data
            extra_prompt_few_shot = ""
            for i, row in train.sample(5).iterrows():
                extra_prompt_few_shot += (
                    f"Example {i+1}\nTexts:\n1. {row['post1']}\n2. {row['post2']}\n"
                    f"{load_trait(personality_trait)} level: {row[personality_trait]}\n\n").replace('_x000D_', '')
                
            extra_prompt_academic_desc_traits = {
    "e": '''High Extraversion (Extroverts): Extroverts tend to use social, outgoing, and upbeat language. They frequently mention other people and social activities, and they express positive emotions openly. For example, a large Facebook study found extraverts were more likely to mention social words and affectionate phrases – “party”, “love you”, “girls”, “ladies”, etc. Similarly, extroverts use more positive emotion words overall and even make more agreements or compliments in text compared to introverts. These patterns reflect an energetic, friendly style (e.g. “Had a great night with everyone – love you all!”).

Low Extraversion (Introverts): Introverts’ language is more reserved and inward-focused. They are less inclined to talk about large social gatherings; instead, they reference solitary or intellectual activities and often use more tentative or introspective tone. Research confirms that introverts are more likely to mention solitary pursuits like “computer”, “Internet”, “reading” in their posts. They also tend to use more self-reflective and anxiety-related words – one study noted that “introverts show more insight and higher rates of anxiety” in their language compared to extraverts. An introvert’s post might thus be more thoughtful or subdued (e.g. reflecting on a quiet night or personal observation rather than hyping a party).

Moderate Extraversion (Ambiverts): Those in the mid-range of extraversion display a mix of both tendencies in language, without extreme markers. Their posts might sometimes be social and positive but other times reserved, depending on context. Because the linguistic differences between introverts and extroverts, while consistent, are relatively small (e.g. extraversion correlates only r≈0.07 with using more positive words), a person with moderate extraversion will not consistently show strong extroverted or introverted cues. In other words, ambiverts’ language may appear balanced, neither overly exuberant nor particularly withdrawn, with social references and personal musings appearing in more equal measure.''',

    "o":  '''High Openness: Individuals high in openness use creative, imaginative, and intellectual language. Their writing often includes references to art, ideas, and novelty. For example, highly open people talk about artistic and creative interests – words around “music”, “art”, “writing” – as well as abstract or contemplative themes like “dream”, “universe”, “soul” appear frequently in their vocabulary. One large blog analysis found that Openness correlated strongly with words related to intellectual and cultural topics (e.g. “poet”, “culture”, “literature”, “art”). They also tend to use more complex or uncommon words. In fact, Pennebaker & King (1999) observed that “openness was positively related to the use of articles and long words” – essentially, more linguistic complexity in their sentences. A highly open person might write a rich, descriptive post (for instance, imaginatively describing a gathering or musing philosophically in a birthday message).

Low Openness: People low in openness (more practical or conventional) favor simple, concrete, and familiar language. Their social media posts are less likely to delve into imaginative themes and more likely to stick to routine or straightforward expressions. They often use more common words and may prefer slang or shorthand over elaborate description. Indeed, one study noted that those low in openness tend to use more abbreviations and Internet shorthand (e.g. “2day” for “today”, “ur” for “your”) in their posts – reflecting a preference for efficiency over florid expression. Overall, a less-open individual’s writing is direct and focuses on real, here-and-now topics rather than abstract ideas. For example, their post about a friend gathering might simply state what happened, without imaginative embellishment.

Moderate Openness: With a moderate level of openness, one’s language is neither especially florid nor especially plain. These individuals might show some curiosity and occasional creative phrasing, but not consistently. Linguistically, they would use a blend of common vocabulary and the occasional complex or unusual term. Empirical evidence suggests a continuum: “writing complexity predicts the level of openness” – so moderately open people demonstrate medium complexity (using some descriptive details and proper grammar, but not as many exotic words as a highly open person). Their social media posts might strike a balance, being clear and communicative with a mild sprinkle of creativity or personal insight when the topic inspires it.''',

    "c":  '''High Conscientiousness: Highly conscientious individuals (organized, responsible) use a careful, positive, and controlled language style. They avoid impulsive or negative expressions. In text analysis, conscientiousness shows a negative relationship with negations and negative emotion words, and a positive relationship with positive emotion words. In other words, conscientious people rarely use words like “not”, “never” or overt negative tones (complaints, anger, sadness) in their posts; instead, they are slightly more inclined to use upbeat, optimistic words. This aligns with the notion that they are self-disciplined and polite in communication. For example, rather than ranting “I’m so upset I forgot your birthday, it was a mess,” a conscientious person might focus on the positive (“Happy birthday! I’m grateful for you and wishing you the best”). They also tend to mention achievement or work-related topics in a constructive way. (Notably, some studies find conscientiousness correlates with words about accomplishment or work, though results can vary) Overall, their posts come across as orderly and positive in tone.

Low Conscientiousness: Those low in conscientiousness (more impulsive, careless) often exhibit the opposite linguistic tendencies. They are more prone to use negative and emotionally charged language, including expressions of anger, frustration, or even profanity that more conscientious people would filter out. In fact, the aforementioned finding implies that if high conscientiousness avoids “negative emotion words”, low conscientiousness is associated with greater use of negative emotion and negation in text. Such individuals may post rants or let their streams-of-thought flow unedited. For example, a low-conscientiousness person might write, “I can’t believe how screwed up this situation is – I forgot my best friend’s birthday, I’m such an idiot, ugh.” This unguarded style often includes more swear words or hostile phrasing (since conscientiousness is linked to lower swearing, the inverse is true for less conscientious folks). In summary, low conscientiousness language is more impulsive, negative, and disorganized.

Moderate Conscientiousness: With an average level of conscientiousness, a person’s writing style is somewhere in between rigidly polished and utterly careless. They likely maintain basic politeness and coherence but won’t be as consistently meticulous or upbeat as a highly conscientious person. Because personality-language correlations are modest in size (often only around 5–10% of variance), a moderately conscientious author might not stand out strongly. Their posts might sometimes contain a minor apology or a casual slang, but generally nothing extreme. In practical terms, moderate conscientiousness yields a fairly neutral style – the person filters some of their thoughts (especially in formal or public posts) but may still occasionally drop a mild complaint or a casual “oops”. They try to be responsible in wording but not to a perfectionist degree.''',

    "a":'''High Agreeableness: Highly agreeable people (trusting, friendly, and cooperative) use warm, polite, and prosocial language. They strive to get along with others, which is reflected in positive word choices and a lack of antagonism in text. Linguistic analyses show that agreeableness correlates positively with words indicating social closeness and positive emotion – for example, frequent use of first-person plural pronouns (“we”, “us”), family and friend terms, and upbeat emotion words (happy, love, nice). At the same time, agreeableness is negatively correlated with anger words and with swear words. In effect, agreeable individuals rarely curse or use hostile language; instead their posts might include supportive comments, compliments, and expressions of gratitude or affection. For instance, an agreeable person wishing a significant other happy birthday might write a kind message praising their partner, and in a reunion post they would emphasize how wonderful it was to be together (avoiding any teasing or negative jokes that could offend). Their overall tone is positive, affirming, and conflict-avoidant.

Low Agreeableness: Low agreeableness (i.e. more disagreeable or antagonistic personalities) is marked by a more critical, aggressive, or negative communication style. These individuals are more willing to use harsh language, including sarcasm, criticism, and profanity. In linguistic terms, they show higher usage of the very categories agreeable people avoid – more anger-related words and more profanity appear in their posts. A disagreeable person’s social media post might be blunt or confrontational (e.g. openly expressing annoyance or making an insult/joke at someone’s expense). They are less inclined to use niceties or positive platitudes. For example, instead of “Had a lovely time, you all are the best,” a low-agreeableness individual might say something like “Yeah, we hung out. Not a big deal.” or even complain about something. In sum, their language can be more combative or uncensored, reflecting a lack of concern for maintaining harmony.

Moderate Agreeableness: People with an average level of agreeableness tend to use generally polite and normal language, without extreme warmth or extreme abrasiveness. They likely balance positive and negative expressions depending on the situation. Such a person might be friendly in their posts most of the time, though not effusively so, and occasionally express disagreement or mild criticism in socially appropriate ways. Given that strong linguistic markers (either very high praise or frequent swearing) are mostly observed at the extremes of agreeableness, a moderately agreeable individual’s text would not be as easily identifiable. Their posts would come off as reasonably civil and cooperative, except when provoked. (Indeed, since language indicators of personality often have small effect sizes, moderate agreeableness may not produce pronounced signals beyond a generally neutral-positive tone.)''',

    "n": '''High Neuroticism: Individuals high in neuroticism (prone to anxiety, moodiness, emotional instability) characteristically use more negative and emotionally intense language. They tend to express worry, frustration, sadness, and other negative feelings in their writing. Studies consistently find that neuroticism correlates positively with the use of negative emotion words – including words related to anxiety, anger, and sadness. For example, highly neurotic persons are more likely to pepper their posts with terms like “hate”, “angry”, “worried”, or “depressed”. In a large social media analysis, neurotic individuals indeed used phrases such as “I hate…” or “sick of…” far more often, whereas emotionally stable people did not. Another known marker is a greater use of first-person singular pronouns (“I”, “me”), reflecting self-focus; more frequent use of first-person pronouns has been associated with higher neuroticism. All this means a neurotic person’s posts often read like emotional outlets – for instance, “I’m so stressed out, I can’t stand it” or “I feel lonely and upset right now”. The language is laden with personal worries, negative evaluations, or complaints.

Low Neuroticism (Emotional Stability): Emotionally stable individuals use a calmer, more positive language in comparison. They experience fewer negative emotions, and their writing reflects this equilibrium. Their posts are less saturated with negative words and more likely to include positive or neutral content. In the earlier example analysis, while high-neurotic folks were saying “hate” and “sick of”, the emotionally stable were talking about enjoyable things – words like “blessed”, “vacation”, “beach”, “team” and other upbeat or social activities came up frequently for low-neurotic (stable) personalities. In general, emotionally stable people’s social media updates sound optimistic or at least even-keeled (for example: “What a great day out with friends, feeling grateful”). They seldom rant or fixate on what’s wrong. Instead of anxiety or anger, they convey contentment and frequently use positive expressions (e.g. expressing gratitude, talking about hobbies or daily life without dramatic negativity). Any frustrations are likely mentioned in a light, controlled manner rather than an emotional outpouring.

Moderate Neuroticism: At moderate levels of neuroticism, the emotional tone of language is mixed. These individuals will sometimes express negative feelings, but not as persistently or intensely as someone high in neuroticism. A moderately neurotic person might occasionally vent (“Ugh, today was annoying”) but also display balance by posting positive or neutral content. Since only high neuroticism strongly increases negative word use, an average person in this trait will show fewer obvious cues. Their posts might alternate – e.g. a normal mix of some complaints when justified, but also plenty of everyday positive moments. In essence, moderate neuroticism yields a variable linguistic style: mostly stable with isolated spikes of negativity under stress. They do not have the constant “doom and gloom” of a highly neurotic person, but they aren’t as unfailingly upbeat as a highly stable person either, falling in the middle with a realistic range of emotions in language.'''
    }

# -------------------------------------------------------------------------------------------------
            # Get most helpful rules using validation data
            # print('min_matches:', len(val['p'])/2)
            # result_val_rules = recursive_rule_filtering(personality_trait, val, rules, 
            #                                             rep, fold, len(val['p'])/2, 0.67,
            #                                             4, path, directory_names)
            
            print('Begginning classification')
            rep_result_test = classifier_using_rules(personality_trait, test, rules, extra_prompt_few_shot, extra_prompt_academic_desc_traits)
            rep_result_test.to_csv(f"{path}{directory_names[1]}/classification results rules, rep {rep} fold {fold['fold']}.csv", index=False)
            rep_results_test.append(rep_result_test) 
        df_rep_results_test = pd.concat(rep_results_test, ignore_index=True)
        df_rep_results_test.to_csv(f"{path}{directory_names[1]}/rep {rep} classification results rules.csv", index=False)
        all_reps.append(df_rep_results_test)
        
        # Rules
        rep_accuracy_rules = df_rep_results_test['score_rules']
        # rep_mae_rules = np.round(mean_absolute_error(df_rep_results_test['true_label'].map(dict_class_to_code), 
        #                          df_rep_results_test['pred_rule_based'].map(dict_class_to_code)), 3)
        all_reps_accuracies_rules.append(rep_accuracy_rules)
        # all_reps_maes_rules.append(rep_mae_rules)

        
        # Few
        rep_accuracy_few = df_rep_results_test['score_few']
        # rep_mae_few = np.round(mean_absolute_error(df_rep_results_test['true_label'].map(dict_class_to_code), 
        #                          df_rep_results_test['pred_few_shot'].map(dict_class_to_code)), 3)
        all_reps_accuracies_few.append(rep_accuracy_few)
        # all_reps_maes_few.append(rep_mae_few)
        

        # Zero
        rep_accuracy_zero = df_rep_results_test['score_zero']
        # rep_mae_zero = np.round(mean_absolute_error(df_rep_results_test['true_label'].map(dict_class_to_code), 
        #                           df_rep_results_test['pred_zero_based'].map(dict_class_to_code)), 3)
        all_reps_accuracies_zero.append(rep_accuracy_zero)
        # all_reps_maes_zero.append(rep_mae_zero)

                
    df_all_reps = pd.concat(all_reps, ignore_index=True)
    df_all_reps.to_csv(f"{path}{directory_names[1]}/overall classification results rules.csv", index=False)
    
    print(f'After {num_reps} replications\n    Rules Accuracy: {np.round(np.mean(all_reps_accuracies_rules), 3)}\n    Standard deviation: {np.round(np.std(all_reps_accuracies_rules), 3)}')
          # \nMean Absolute Error:    {np.round(np.mean(all_reps_maes_rules), 3)}\n
    print(f'After {num_reps} replications\n    Few Accuracy: {np.round(np.mean(all_reps_accuracies_few), 3)}\n    Standard deviation: {np.round(np.std(all_reps_accuracies_few), 3)}')
          # Mean Absolute Error:    {np.round(np.mean(all_reps_maes_few), 3)}\n
    print(f'After {num_reps} replications\n    Zero Accuracy: {np.round(np.mean(all_reps_accuracies_zero), 3)}\n    Standard deviation: {np.round(np.std(all_reps_accuracies_zero), 3)}')
          # \nMean Absolute Error:    {np.round(np.mean(all_reps_maes_zero), 3)}

    plot_confusion_matrix(y_true = df_all_reps['true_label'], y_pred = df_all_reps['pred_rule_based'], labels = ['High', 'Moderate', 'Low'], title = 'Rule based Model Results', save_path=f'{path}{directory_names[1]}/Rule-based Model Results')
    plt.clf()  # clear current figure
    plot_confusion_matrix(y_true = df_all_reps['true_label'], y_pred = df_all_reps['pred_few_based'],   labels = ['High', 'Moderate', 'Low'], title = 'Few Shot Model Results', save_path=f'{path}{directory_names[1]}/Few Shot Model Results')
    plt.clf()  # clear current figure
    plot_confusion_matrix(y_true = df_all_reps['true_label'], y_pred = df_all_reps['pred_zero_based'],  labels = ['High', 'Moderate', 'Low'], title = 'Zero Shot Model Results', save_path=f'{path}{directory_names[1]}/Zero Shot Model Results')

if __name__ == "__main__":

    
    try:
        start = datetime.now()
        num_reps=10
        # print(load_posts_and_trait_true_label('e'))
        main(personality_trait = 'e', num_reps=num_reps)

    # Temperature is not relevant in gpt-5 because it is a reasoning model.
        # main(personality_trait = 'e', num_reps=num_reps, temp = 'warm')
        # main(personality_trait = 'e', num_reps=num_reps, temp = 'hot')
        # main(personality_trait = 'e', num_reps=num_reps, temp = 'standard and warm')
        # main(personality_trait = 'e', num_reps=num_reps, temp = 'standard and hot')
        # main(personality_trait = 'e', num_reps=num_reps, temp = 'warm and hot')
        # main(personality_trait = 'e', num_reps=num_reps, temp = 'all')
        print(f'\nTime elapsed: {datetime.now() - start}')

        # Notify on success
        notification.notify(
            title="✅ Code Completed",
            message="The script finished successfully.",
            timeout=5)
    except Exception as e:
        # Notify on error
        error_message = f"🚨 ERROR: {type(e).__name__} - {str(e)}"
        notification.notify(
            title="❌ Code Crashed",
            message=error_message,
            timeout=10
        )
        print(f'\nTime elapsed: {datetime.now() - start}')

        # Optional: print full traceback to console or file
        traceback.print_exc()
        # You can also log it to a file if needed:
        with open("C:/Users/User/OneDrive/Desktop/Masters/thesis/code 3 classes/error_log.txt", "w") as f:
            f.write(traceback.format_exc())  # main_recursive()
    end = datetime.now()
    end_txt = end.strftime("%Y-%m-%d_%H-%M-%S")
    print('END', end_txt)