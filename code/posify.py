# proteinMusic.py
#
# Demonstrates how to utilize list operations to build an unfolding piece
# of music based on the first 13 amino acids in the human thymidylate
# synthase A (ThyA) protein.
#
# The piece starts with the 1st amino acid, continues with the 1st and 2nd
# amino acids, then with the 1st, 2nd, and 3rd amino acids, and so on,
# until all 13 amino acids have been included.
#
# See: Takahashi, R. and Miller, J.H. (2007), "Conversion of Amino-Acid
# Sequence in Proteins to Classical Music: Search for Auditory Patterns",
# Genome Biology, 8(5), p. 405. 
 
import os
from music import *
import random
import json
import colorsys

def chord_const(chord, key1, base=48, m="major", sevenths=True): # m : 'major' or 'minor
   notelist = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
   if m == "major":
      chordroot_dict = {1:0, 2:2, 3:4, 4:5, 5:7, 6:9, 7:11}
   elif m == "minor":
      chordroot_dict = {1:0, 2:2, 3:3, 4:5, 5:7, 6:8, 7:10}
      
      
   chordrootnumb = int(notelist.index(key1)) + int(chordroot_dict[chord])
   root = chordrootnumb + base
   maj3rd = root + 4
   min3rd = root + 3
   fifth = root + 7
   dimfifth = root + 6
   maj7th = root + 11
   min7th = root + 10

   if m == "major":
      if chord== 1 or chord == 4:
         ret = [root, fifth, maj3rd, maj7th]
      elif chord == 2 or chord == 3 or chord == 6:
         ret = [root, fifth, min3rd, min7th]
      if chord== 5:
         ret = [root, fifth, maj3rd, min7th]
      elif chord == 7:
         ret = [root, dimfifth, min3rd, min7th]
            
   elif m == "minor":
      if chord== 1 or chord== 4 or chord== 5:
         ret = [root, fifth, min3rd, min7th]
      elif chord== 3 or chord== 6:
         ret = [root, fifth, maj3rd, maj7th]
      elif chord== 7:
         ret = [root, fifth, maj3rd, min7th]
      elif chord == 2:
         ret = [root, dimfifth, min3rd, min7th]

   if sevenths == False:
      return ret[:-1]
   return ret

def getDrumPart(startTime):
 
   # 35% of the time we try something different
   comeAlive = 0.35
   
   # how many measures to play
   measures = 1   
   
   bassDrumPhrase = Phrase(startTime)     # create phrase for each drum sound
   snareDrumPhrase = Phrase(startTime)
   
   ##### create musical data
   # kick
   # bass drum pattern (one bass 1/2 note) x 2 = 1 measure
   # (we repeat this x 'measures')
   for i in range(2 * measures):
   
      dynamics = random.randint(30, 40)   # add some variation in dynamics
      n = Note(ACOUSTIC_BASS_DRUM, HN, dynamics)
      bassDrumPhrase.addNote(n)
   
   # snare
   # snare drum pattern (one rest + one snare 1/4 notes) x 2 = 1 measure
   # (we repeat this x 'measures')
   for i in range(2 * measures):
   
      r = Note(REST, QN)
      snareDrumPhrase.addNote(r)
   
      dynamics = random.randint(30, 40)    # add some variation in dynamics
      n = Note(SNARE, QN, dynamics)
      snareDrumPhrase.addNote(n)

   return bassDrumPhrase, snareDrumPhrase

# random.seed()
meta = {
   'M1' : [0],
   'M2' : [0],
   'M3' : [0],
   'M4' : [0],
   'M5' : [0],
   'S1' : [0],
   'S2' : [1],
   'S3' : [0],
   'S4' : [0],
   'S5': [0],
}

def openJSON(fjson):
   myFile=open(fjson, 'r')
   myObject=myFile.read()
   u = myObject.decode('utf-8-sig')
   myObject = u.encode('utf-8')
   myFile.encoding
   myFile.close()
   return json.loads(myObject,'utf-8')

def weighted_choice(seq, weights):
   assert len(seq) == len(weights)

   total = sum(weights)
   r = random.uniform(0, total)
   upto = 0
   for i in range(len(seq)):
      if upto + weights[i] >= r:
            return seq[i]
      upto += weights[i]
   assert False, "Shouldn't get here"

key_list = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
circle_of_fifth = ['C','G','D','A','E','B','F#','Db','Ab','Eb','Bb','F']

for nn in meta.keys():
   pitches, melodies = [], []
   k=meta[nn][0]
   
   mj = openJSON('eval_0802/output/{}.json'.format(nn)) 
   hue_counts = mj['hue_counts']
   hueList = mj['hue_original_img']
   mask=mj['human_{}'.format(k)]['mask']
   tempo= mj['human_{}'.format(k)]['tempo'] - 10

   
   s, ss, si = 0., 0., 0.
   for i, x in enumerate(hue_counts):
      s+=x

      if ss < hue_counts[i]:
         ss = hue_counts[i]
         si = i
      # ss+=hueList[i][0]

   hue_avg = hueList[si][0] #ss / len(hue_counts)
   hue_prob = [x / s for x in hue_counts]



   # set of pitches/rhythms to use for building incremental piece
   
   key_idx = int ( hue_avg * len(circle_of_fifth) )

   sod_density = True
   sod_tempo = True

   if sod_density and sod_tempo:
      tag = 'C_sod-all_v2'
   elif not sod_density and sod_tempo:
      tag = 'B_sod-tempo_v2'
   elif not sod_density and not sod_tempo:
      tag = 'A_cof_random'

   score  = Score("first")  # 110 bpm
   dname = 'eval_0802/{}'.format(tag)
   if not os.path.exists(dname):
      os.makedirs(dname)#, exist_ok=True)
   fname_midi = '{}/{}.midi'.format(dname,nn)

   chord = 1
   n=16
   # add incremental phrases to a part
   chord_part = Part(PIANO, 0)
   melody_part = Part(HARP, 1)
   drumsPart = Part("Drums", 0, 9)  # using MIDI channel 9 (percussion)
   st = 0
   for j in range(len(hueList)):
      if j > 0:
         st += 4 * 60. / tempo
      h, l, s = weighted_choice(hueList, hue_prob)
      chord = (0 + int(h*7)) % 7 + 1
      m = "minor" if s < 0.5 else "major"

      r,g,b = colorsys.hls_to_rgb(h,l,s)

      if not sod_tempo:
         tempo = mapValue(int(r*256), 0, 255, 30, 150)
      
      # if s < 0.1: 
      #    key_idx = (key_idx + 1) % 12
      
      p = chord_const(chord, circle_of_fifth[key_idx], base=48, m=m)

      if not sod_density:
         n = random.sample( list(range(4,16)), 1 )[0]
      notes = p + [p[0]+12]
      notes = notes + notes[::-1]
      notes = notes *8
      notes = [notes[0]] + random.sample(notes, n-2) + [notes[-1]]

      if sod_density:
         if j > 0 and j%4 == 0:
            mask = random.sample(mask, len(mask)) # permute
      else:
         mask = [1]*n + [0] * (16-n)
         mask = random.sample(mask, len(mask)) # permute


      mm = [REST if x == 0 else 0 for x in mask]
      mn = [notes[i] * mask[i] for i in range(n)]
      
      melody_bar = [mm[i] + mn[i] for i in range(n)]
      melody_dur = [4. / n] * n

      melody_phrase = Phrase(st)
      melody_phrase.addNoteList(melody_bar, melody_dur)

      chord_bar = [p[:-1], REST, p[:-1], REST, p[:-1], REST]
      chord_dur = [1./4,  9./4, 0.5/4, 2.5/4, 0.5/4, 2.5/4]
      chord_phrase = Phrase(st)            # create empty phrase
      chord_phrase.addNoteList(chord_bar, chord_dur)

      bassDrumPhrase, snareDrumPhrase = getDrumPart(st)

      melody_phrase.setTempo(tempo)
      chord_phrase.setTempo(tempo)
      bassDrumPhrase.setTempo(tempo)
      snareDrumPhrase.setTempo(tempo)

      melody_part.addPhrase( melody_phrase )
      chord_part.addPhrase ( chord_phrase )
      drumsPart.addPhrase(bassDrumPhrase)
      drumsPart.addPhrase(snareDrumPhrase)

   # Play.midi( score)
   melody_part.setDynamic(30)
   chord_part.setDynamic(25)
   drumsPart.setDynamic(25)
   # Mod.elongate(drumsPart,2)

   score.addPart(melody_part)
   score.addPart(chord_part)
   score.addPart(drumsPart)

   Write.midi(score, fname_midi)
   # break

exit()