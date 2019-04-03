
import nltk, re
strange_string = """Strange women lying in ponds distributing swords is no basis for a system of government ."""
strange_words = strange_string.split()

######################################################################
# Question 1

print "* Question 1 a"
print strange_words[1:6:2]
print

print "* Question 1 b"
print [len(w) for w in strange_words if w.endswith("s")]
print

print "* Question 1 c"
print re.sub(r".([egnst]+ )", r"-\1", strange_string)
print

print "* Question 1 d"
print nltk.FreqDist(w[-1] for w in strange_words).hapaxes()
print

######################################################################
# Question 2

# Solution with a list comprehension
def shorten_long_words_1(words):
    return [w[:2] + w[-2:] for w in words]

# Solution with a for loop
def shorten_long_words_2(words):
    result = []
    for w in words:
        result.append(w[:2] + w[-2:])
    return result

print "* Question 2"
print shorten_long_words_1(strange_words)
print shorten_long_words_2(strange_words)
print
# ['Stge', 'woen', 'lyng', 'pods', 'ding', 'swds', 'bais', 'syem', 'gont']

######################################################################
# Question 3

# Solution with a regexp
def double_vowel_1(string):
    return re.sub(r"([aeiouy])", r"\1l\1", string)

# Solution with a for loop
def double_vowel_2(string):
    result = ""
    for char in string:
        if char in "aeiouy":
            result += char + "l" + char
        else:
            result += char
    return result

print "* Question 3"
print double_vowel_1(strange_string)
print double_vowel_2(strange_string)
print
# 'Stralangele wolomelen lylyiling ilin polonds dilistrilibulutiling swolords ilis nolo balasilis folor ala sylystelem olof golovelernmelent .'

######################################################################
# Question 4

IOBcorpus = """
Strange/B women/I lying/O in/O ponds/B distributing/I swords/I is/O no/B basis/I for/O
a/B system/I of/O government/B ./O Supreme/O executive/B power/I derives/O from/O
a/B mandate/I from/I the/I masses/I ,/O not/O from/O some/O farcical/O aquatic/O ceremony/O ./O
You/B can't/O expect/O to/O wield/O supreme/O executive/B power/I just/O 'cause/O
some/B watery/I tart/I threw/O a/B sword/I at/I you/I !/O
"""

Btags = IOBcorpus.count("/B")
Itags = IOBcorpus.count("/I")
Otags = IOBcorpus.count("/O")

print "* Question 4"
print "nr B tags:", Btags
print "nr I tags:", Itags
print "nr O tags:", Otags
print "Total words:", Btags + Itags + Otags
print

######################################################################
# Question 5

true_positives  = ["Strange women", "no basis", "a system", "government", "You", "some watery tart"]
false_positives = ["ponds distributing swords", "executive power", "a mandate from the masses", "executive power", "a sword at you"]
false_negatives = ["ponds", "swords", "Supreme executive power", "a mandate", "the masses",
                   "some farcical aquatic ceremony", "supreme executive power", "a sword", "you"]

TP = len(true_positives)
FP = len(false_positives)
FN = len(false_negatives)

print "* Question 5"
print "True Positives: ", TP
print "False Positives:", FP
print "False Negatives:", FN
print "Precision = TP / (TP+FP) = %d / %d" % (TP, TP + FP)
print "Recall    = TP / (TP+FN) = %d / %d" % (TP, TP + FN)
print

######################################################################
# Questions 6+7

grammar = nltk.parse_cfg("""
S -> NP VP DL
VP -> VBZ NP | VP ADVL
NP -> DT NN | NN | JJ NN | NP ADVL
ADVL -> IN NP | VBG NP | VBG IN NP

JJ -> "Strange"
DT -> "no" | "a"
NN -> "women" | "ponds" | "swords"
NN -> "basis" | "system" | "government"
VBZ -> "is"
VBG -> "lying" | "distributing"
IN -> "in" | "for" | "of"
DL -> "."
""")

parser = nltk.ChartParser(grammar)
alltrees = parser.nbest_parse(strange_words)
strange_tree = alltrees[0]

print "* Questions 6+7"
print "Nr. of trees:", len(alltrees)
print "The first tree:"
print strange_tree
print
print "strange_tree[1][1][0].leaves() =", strange_tree[1][1][0].leaves()
print
print "The second tree:"
print alltrees[1]
print
print "The last tree:"
print alltrees[-1]
