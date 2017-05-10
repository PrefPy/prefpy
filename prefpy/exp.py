import copy
import prefpy_io
import itertools
import math
import json
import os
from preference import Preference
from profile import Profile
import mechanism

# Below is a template Main which shows off some of the
# features of this library.
if __name__ == '__main__':
    # Grab and read a file.
    # os.chdir('D:\Social Choice\data\soc-3-hardcase')
    os.chdir('D:\\Social Choice\\data\\toc')
    # inputfile = input("Input File: ")
    inputfile = "ED-00006-00000038.toc"
    # inputfile = "M30N30-233.csv"
    inf = open(inputfile, 'r')
    cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
    print("cmap=",cmap)
    print("rmaps=",rmaps)
    print("rmapscounts=",rmapscounts)
    print("nvoters=",nvoters)
    profile = Profile(cmap, preferences=[])
    Profile.importPreflibFile(profile, inputfile)
    print(profile.getOrderVectors())
    print(profile.getPreferenceCounts())
    ordering = profile.getOrderVectors()
    rankmaps = profile.getRankMaps()
    print("rankmaps=",rankmaps)
    print(min(ordering[0]))
    # if(min(ordering[0])==0):
    #     print("ok")
    # else:
    #     print("not ok")

    stvwinners = mechanism.MechanismSTV().STVwinners(profile)
    print("stvwinners=", stvwinners)
    baldwinners = mechanism.MechanismBaldwin().baldwin_winners(profile)
    print("baldwinners=", baldwinners)
    coombswinners = mechanism.MechanismCoombs().coombs_winners(profile)
    print("coombswinners=", coombswinners)