from itertools import count
import random
import pandas as pd
import nltk
nltk.download('words')
from nltk.corpus import words

NUM_PLAINTEXTS = 1000
KEY_AMOUNT = 100


# The length of cirpher text must large enough to learn the encrypt structure
CHARSET_THRESHOLD = 625 * 3
def create_matrix(key):
    """
    Create 5x5 matrix from key
    """

    key = key.upper()
    matrix = [[0 for i in range(5)] for j in range(5)]
    letters_added = []
    row = 0
    col = 0
    for letter in key:
        if letter not in letters_added:
            matrix[row][col] = letter
            letters_added.append(letter)
        else:
            continue
        if col == 4:
            col = 0
            row += 1
        else:
            col += 1
    for letter in range(65, 91):
        if letter == 74:
            continue
        if chr(letter) not in letters_added:
            letters_added.append(chr(letter))
    index = 0
    for i in range(5):
        for j in range(5):
            matrix[i][j] = letters_added[index]
            index += 1
    return matrix

def separate_same_letters(message):
    """
    Same letter in plain text should be separate by "X"
    Append "X" if message text is odd length
    """
    index = 0
    while index < len(message):
        l1 = message[index]
        if index == len(message) - 1:
            if l1 == 'X':
                if message.count('X') % 2 != 0:
                    message = message[:-1]
            else:
                message = message + 'X'
            index += 2
            continue
        l2 = message[index + 1]
        if l1 == l2:
            message = message[:index + 1] + "X" + message[index + 1:]
        index += 2
    return message

def index_of(letter, matrix):
    """
    Return x, y position of letter in cirpher matrix
    """

    for i in range(5):
        for j in range(5):
            if matrix[i][j] == letter:
                return i, j
    return -1, -1

def playfair(key, message, encrypt=True):
    """
    Encrypt/Decrypt controller
    """

    inc = 1
    # Reverse direction for encrypt
    if encrypt == False:
        inc = -1
    matrix = create_matrix(key)
    message = message.upper()
    message = message.replace(' ', '')
    message = separate_same_letters(message)
    cipher_text = ''
    for (l1, l2) in zip(message[0::2], message[1::2]):
        row1, col1 = index_of(l1, matrix)
        row2, col2 = index_of(l2, matrix)
        # Case 1: Same row → shift right (encryption) or left (decryption)
        if row1 == row2:
            cipher_text += matrix[row1][(col1 + inc) % 5] + matrix[row2][(col2 + inc) % 5]
        # Case 2: Same column → shift down (encryption) or up (decryption)
        elif col1 == col2:
            cipher_text += matrix[(row1 + inc) % 5][col1] + matrix[(row2 + inc) % 5][col2]
        # Case 3: Rectangle rule → swap column positions
        else:
            cipher_text += matrix[row1][col2] + matrix[row2][col1]



    # Remove inserted 'X' from decryption
    if not encrypt:
        cipher_text = cipher_text.replace('X', '')
    return cipher_text

def generate_plaintexts(num_plaintexts):
    """
    Generate plain text sample
    """
    # plaintexts = []

    word_set = words.words()
    random.shuffle(word_set)
    word_set_len = len(word_set)
    counter = 0
    for _ in range(num_plaintexts):
        plaintext = ""
        while len(plaintext) < CHARSET_THRESHOLD:
            plaintext += word_set[counter %  word_set_len]
            if counter == word_set_len - 1:
                random.shuffle(word_set)
                counter = 0
            counter += 1
        yield plaintext
        # plaintexts.append(plaintext)
    # return plaintexts



def generate_keys(thres_hold = KEY_AMOUNT):
    """
    Generate plain text sample
    """
    word_set = words.words()

    word_set_len = len(word_set)
    random.shuffle(word_set)
    for i in range(thres_hold):
        word = word_set[i %  word_set_len]
        if i == word_set_len - 1:
            random.shuffle(word_set)
        yield word[:10].upper()

if __name__ == '__main__':
    plaintexts = generate_plaintexts(NUM_PLAINTEXTS)
    keys  = list(generate_keys())
    # key = 'SECRET'
    data = {'Plain Text': [], 'Key':[],  'Encrypted Text': []}
    max_char_len = 0

    for i, plaintext in enumerate(plaintexts):
        key = random.choice(keys)
        encrypted_text = playfair(key, plaintext)
        data['Plain Text'].append(plaintext)
        data['Key'].append(key)
        data['Encrypted Text'].append(encrypted_text)
        max_char_len = max(max_char_len, len(encrypted_text))
        print(plaintext,  " => " , encrypted_text)
    df = pd.DataFrame(data)
    print("Max character len of encrypted text ", max_char_len)
    # df.to_excel('PLAYFAIR_CIPHER_DATASET.xlsx', index=False)
    df.to_excel(f'PLAYFAIR_CIPHER_DATASET_RANDOM_KEY_{NUM_PLAINTEXTS}.xlsx', index=False)
