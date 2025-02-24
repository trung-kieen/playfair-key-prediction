import random
import pandas as pd
import nltk
nltk.download('words')
from nltk.corpus import words

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
    for _ in range(num_plaintexts):
        plaintext = random.choice(words.words())
        yield plaintext
        # plaintexts.append(plaintext)
    # return plaintexts



def generate_keys(thres_hold = 100):
    """
    Generate plain text sample
    """
    for _ in range(thres_hold):
        word = random.choice(words.words())
        if len(word) > 5:
            yield word[:5].upper()

if __name__ == '__main__':
    num_plaintexts = 10000
    plaintexts = generate_plaintexts(num_plaintexts)
    keys  = list(generate_keys())
    # key = 'SECRET'
    data = {'Plain Text': [], 'Key':[], 'Cipher Text': [], 'Decrypted Text': []}

    for i, plaintext in enumerate(plaintexts):
        key = random.choice(keys)
        ciphertext = playfair(key, plaintext)
        decrypted_text = playfair(key, ciphertext, encrypt=False)
        data['Plain Text'].append(plaintext)
        data['Key'].append(key)
        data['Cipher Text'].append(ciphertext)
        data['Decrypted Text'].append(decrypted_text)
        print(plaintext)
    df = pd.DataFrame(data)
    # df.to_excel('PLAYFAIR_CIPHER_DATASET.xlsx', index=False)
    df.to_excel('PLAYFAIR_CIPHER_DATASET_RANDOM_KEY.xlsx', index=False)
