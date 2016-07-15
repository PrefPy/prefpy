'''
	File: 	io.py
	Author:	Nicholas Mattei (nicholas.mattei@nicta.com.au)
	Date:	April 4, 2013
			November 6th, 2013
	
  * Copyright (c) 2014, Nicholas Mattei and NICTA
  * All rights reserved.
  *
  * Developed by: Nicholas Mattei
  *               NICTA
  *               http://www.nickmattei.net
  *               http://www.preflib.org
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions are met:
  *     * Redistributions of source code must retain the above copyright
  *       notice, this list of conditions and the following disclaimer.
  *     * Redistributions in binary form must reproduce the above copyright
  *       notice, this list of conditions and the following disclaimer in the
  *       documentation and/or other materials provided with the distribution.
  *     * Neither the name of NICTA nor the
  *       names of its contributors may be used to endorse or promote products
  *       derived from this software without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY NICTA ''AS IS'' AND ANY
  * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL NICTA BE LIABLE FOR ANY
  * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

About
--------------------
	This file contains a set of useful modules for reading, writing, and converting
	PrefLib files between the various formats.
	
'''

import operator
import itertools
import math
import copy


# Given a candmap and a votemap, write the output in
# Preflib format to the given file.
def write_map(candmap, nvoters, votemap, file):
	#Write the header
	file.write(str(len(candmap.keys())) + '\n')
	#Make Candidate List
	for ele in sorted(candmap.keys()):
		file.write(str(ele) + "," + str(candmap[ele]) + " \n")
	#Write the Number of Voters, Total number of votes and Unique Orders...
	file.write(str(nvoters) + "," + str(sum(votemap.values())) + "," + str(len(votemap.keys())) + "\n")
	#Write the votes.. (sorted by count)
	for vote, count in sorted(votemap.items(), key=lambda x: x[1], reverse=True):
		file.write(str(count) + "," + vote + "\n")
		
# Given a file in one of the Preflib Election Data 
# formats, return a list of rankmaps.
def read_election_file(inputfile):
	#first element is the number of candidates.
	l = inputfile.readline()
	numcands = int(l.strip())
	candmap = {}
	for i in range(numcands):
		bits = inputfile.readline().strip().split(",")
		candmap[int(bits[0].strip())] = bits[1].strip()
	
	#now we have numvoters, sumofvotecount, numunique orders
	bits = inputfile.readline().strip().split(",")
	numvoters = int(bits[0].strip())
	sumvotes = int(bits[1].strip())
	uniqueorders = int(bits[2].strip())
	
	rankmaps = []
	rankmapcounts = []
	for i in range(uniqueorders):
		rec = inputfile.readline().strip()
		#need to parse the rec properly..
		if rec.find("{") == -1:
			#its strict, just split on ,
			count = int(rec[:rec.index(",")])
			bits = rec[rec.index(",")+1:].strip().split(",")
			cvote = {}
			for crank in range(len(bits)): 
				cvote[int(bits[crank])] = crank+1
			rankmaps.append(cvote)
			rankmapcounts.append(count)
		else:
			count = int(rec[:rec.index(",")])
			bits = rec[rec.index(",")+1:].strip().split(",")
			cvote = {}
			crank = 1
			partial = False
			for ccand in bits:
				if ccand.find("{") != -1:
					partial = True
					t = ccand.replace("{","")
					cvote[int(t.strip())] = crank
				elif ccand.find("}") != -1:
					partial = False
					t = ccand.replace("}","")
					cvote[int(t.strip())] = crank
					crank += 1
				else:
				 	cvote[int(ccand.strip())] = crank
				 	if partial == False:
				 		crank += 1
			rankmaps.append(cvote)
			rankmapcounts.append(count)
		
	#Sanity check:
	if sum(rankmapcounts) != sumvotes or len(rankmaps) != uniqueorders:
		print("Error Parsing File: Votes Not Accounted For!")
		exit()
	
	return candmap, rankmaps, rankmapcounts, numvoters

# Given a pairwise map return the weighted and unweighted majority graphs.
# and a boolean for isTournament.
def pairwise_to_relation(candmap, pairwisemap):
	#compute the weighted majority relation...
	majrelation = {}
	isTournament = True
	for cpair in itertools.combinations(candmap.keys(), 2):
		#Write the bigger direction....
		if pairwisemap.get(str(cpair[0])+","+str(cpair[1]), 0) > pairwisemap.get(str(cpair[1])+","+str(cpair[0]), 0):
			majrelation[str(cpair[0])+","+str(cpair[1])] = pairwisemap.get(str(cpair[0])+","+str(cpair[1]), 0) - pairwisemap.get(str(cpair[1])+","+str(cpair[0]), 0)
		elif pairwisemap.get(str(cpair[1])+","+str(cpair[0]), 0) > pairwisemap.get(str(cpair[0])+","+str(cpair[1]), 0):
			majrelation[str(cpair[1])+","+str(cpair[0])] = pairwisemap.get(str(cpair[1])+","+str(cpair[0]), 0) - pairwisemap.get(str(cpair[0])+","+str(cpair[1]), 0)
		else:
			isTournament = False
	unwmaj = {x: 1 for x in majrelation.keys()}
	return majrelation, unwmaj, isTournament
		
# Given a candidate set and a vote map, pad all the votes by placing unranked
# candidates tied at the end of the vote.
def extend_partial_complete(candmap, votemap):
	extended = {}
	#Go through each vote...
	for cvote in votemap.keys():
		#extend the vote with all the non-appearing candidates.
		voted = set()
		#remove any { or } in the list...
		cleanvote = cvote.replace("{","")
		cleanvote = cleanvote.replace("}","")
		for sp in cleanvote.strip().split(","):
			#need to make sure that we break up and partial pieces...
			ranks = sp.strip()
			if (len(ranks.strip()) == 0):
				print("caught")
				print(votemap)
				exit()
			for x in ranks.strip().split(","):
				voted.add(int(x.strip()))
		if len(voted) != len(candmap.keys()):
			tail = ""
			#if the didn't rank more than 1 candidate.
			if len(candmap.keys()) - len(voted) > 1:
				tail = "{"
				for x in candmap.keys():
					if x not in voted:
						tail += str(x) +"," 
				tail = tail[:len(tail)-1]+"}"
			else:
				for x in candmap.keys():
					if x not in voted:
						tail += str(x) 
			
			#pop it on the end...
			extended[cvote+","+tail] = (extended.get(cvote+","+tail, 0) + votemap[cvote])
		else:
			extended[cvote] = (extended.get(cvote, 0) + votemap[cvote])
	
	return extended
	
# Given a set of votes, return the pairwise
# of all the candidates.
def convert_to_pairwise(candmap, votemap):
	#Generate a hash of all pairs of candidates
	pairwisemap = {}
	
	ranklist = []
	#Convert to a rankmap FIRST... not per pair...
	for cvote in votemap.keys():
		#convert vote into candidate --> rank map
		cand_rank ={}
		crank = 0
		for rank in cvote.split(","):
			rank = rank.strip("{} ")
			if len(rank.split(" ")) > 1:
				for cand in rank.split(" "):
					cand = cand.strip("{} ")
					cand_rank[cand] = crank
			else:
				cand_rank[rank] = crank
			crank+= 1
		ranklist.append(cand_rank)
	
	#iterate over all combinations and check both directions.
	for cpair in itertools.combinations(candmap.keys(), 2):
		for cand_rank in ranklist:
			#assign all the votes counted one way or the other if BOTH CANDIDATES APPEAR!
			if str(cpair[0]) in cand_rank.keys() and str(cpair[1]) in cand_rank.keys():
				if cand_rank[str(cpair[0])] < cand_rank[str(cpair[1])]:
					pairwisemap[str(cpair[0])+","+str(cpair[1])] = (pairwisemap.get(str(cpair[0])+","+str(cpair[1]), 0) + votemap[cvote])
				elif cand_rank[str(cpair[1])] < cand_rank[str(cpair[0])]:
					pairwisemap[str(cpair[1])+","+str(cpair[0])] = (pairwisemap.get(str(cpair[1])+","+str(cpair[0]), 0) + votemap[cvote])
		
	return pairwisemap
				
# Given a set of verticies and names, write out the matching
# data file format.
def write_match(vertexmap, edges, file):
	#write the first line...
	file.write(str(len(vertexmap.keys())) + "," + str(len(edges.keys())) + "\n")
	#write the names of the verticies...
	for ele in sorted(vertexmap.keys()):
		file.write(str(ele) + "," + str(vertexmap[ele]) + " \n")
	#write the edges... sorted by numerical first element.
	for ele in sorted(edges.keys(), key=lambda x: int(x.split(",")[0])):
		file.write(str(ele) + "\n")
		
# Pretty printer for an election result.
def pp_result_toscreen(candmap, scores):
	print("\n\n{:^8}".format("n") + "|" + "{:^35}".format('Candidate') + "|" + "{:^35}".format('Score'))
	print("{:-^75}".format(""))	
	for s in sorted(scores, key=scores.get, reverse=True):
		print("{:^8}".format(str(s)) + "|" +"{:^35}".format(str(candmap[s])) + "|" + "{:^35}".format(str(scores[s])))
	return 0
	
# Pretty printer for a profile. Print 
# the preflib format to the screen.
def pp_profile_toscreen(candmap, rankmaps, rankmapcounts):
	#Sort the rankmap/rankmapkey pair based on item frequency...
	srmaps = [k for k, v in sorted(zip(rankmaps, rankmapcounts), key=operator.itemgetter(1), reverse=True)]
	srmapc = [v for k, v in sorted(zip(rankmaps, rankmapcounts), key=operator.itemgetter(1), reverse=True)]	
	
	#pretty print the candidate map.
	print("\n\n{:^8}".format("n") + "|" + "{:^35}".format('Candidate'))
	print("{:-^75}".format(""))	
	for ccand in candmap.keys():
		print("{:^8}".format(str(ccand)) + "|" + "{:^35}".format(str(candmap[ccand])))
	print("{:-^75}".format(""))	
	#print the rank map and counts...
	print("{:^8}".format("Count") + "|" + "{:^35}".format('Profile'))
	for i in range(len(srmapc)):
		outstr = ""
		# Convert rankmap[i] to rorder which is rank --> candi
		rorder = {x:[] for x in srmaps[i].values()}
		for ccand in srmaps[i].keys():
			rorder[srmaps[i][ccand]].append(ccand)
			
		for cr in sorted(rorder.keys()):
			if len(rorder[cr]) > 1:
				#assemble a multivote.
				substr = "{"
				for ccand in rorder[cr]:
					substr += str(ccand) + ","
				outstr += substr[:len(substr)-1] + "},"
			else:
				outstr += str(rorder[cr][0]) + ","
		print("{:^8}".format(str(srmapc[i])) + "|" + "{:^35}".format(str(outstr[:len(outstr)-1])))
	
# Evaluate a vote for a given score vector.
def evaluate_scoring_rule(candmap, rankmaps, rankmapcounts, scorevec):
	if len(scorevec) != len(candmap):
		print("Score Vector and Candidate Vector must have equal length")
		exit()	
	#initialize the score map.
	scores = {x:0 for x in candmap.keys()}
	#for each rank map, for each rank, multiply...
	for i in range(len(rankmaps)):
		for j in rankmaps[i].keys():
			scores[j] += rankmapcounts[i] * scorevec[rankmaps[i][j]-1]
	return scores
	 
# Relabel the candidates according to a given score vector so that 
# the winner of the election is candidate 1.
def relabel(candmap, rankmaps, rankmapcounts, scores):
	
	#basically, take the scores and make a candidate mapping old --> new
	#then copy and modify the candmap and the rankmap... counts are the same...
	cand_remapping = {}
	newnum = 1
	for s in sorted(scores, key=scores.get, reverse=True):
		#highest score candidate goes to 1...
		cand_remapping[s] = newnum
		newnum += 1
		
	re_candmap = {cand_remapping[x]:candmap[x] for x in candmap.keys()}
	
	#same deal for the rankmaps....
	re_rankmaps = []
	for cmap in rankmaps:
		re_rankmaps.append({cand_remapping[x]:cmap[x] for x in cmap.keys()})
	
	return re_candmap, re_rankmaps, rankmapcounts
			
# Relabel the candidates according to the most common complete order.
# the winner of the election is candidate 1.
def max_relabel(candmap, rankmaps, rankmapcounts):
	
	#find the rankmap with the max count AND it's complete...
	relabelorder = 0
	for x in sorted(rankmapcounts, reverse=True):
		if len(rankmaps[rankmapcounts.index(x)]) == len(candmap):
			relabelorder = rankmapcounts.index(x)
	
	#basically, take the scores and make a candidate mapping old --> new
	#such that the most numerous complete vote is the ranking.
	#then copy and modify the candmap and the rankmap... counts are the same...
	cand_remapping = {}
	newnum = 1
	for s in rankmaps[relabelorder].keys():
		#highest score candidate goes to 1...
		cand_remapping[s] = newnum
		newnum += 1
		
	re_candmap = {cand_remapping[x]:candmap[x] for x in candmap.keys()}
	
	#same deal for the rankmaps....
	re_rankmaps = []
	for cmap in rankmaps:
		re_rankmaps.append({cand_remapping[x]:cmap[x] for x in cmap.keys()})
	
	return re_candmap, re_rankmaps, rankmapcounts
		
# Convert a rankmap to an order of candidate number...
def rankmap_to_order(rm):
	order = [-1]*len(rm.keys())
	for i in rm.keys():
		order[rm[i]-1] = i
	return order

# Convert a set of rankmap to be a mapping from Rank --> Candidate
def rankmap_convert_rank_to_candidate(rmaps):
	rank_to_cand = []
	for i in rmaps:
		rank_to_cand.append({v:k for k, v in i.items()})
	return(rank_to_cand)

#Convert a set of rank_to_candidate back to a set of rankmaps.
def rank_to_candidate_convert_to_rankmap(r_to_c):
	r_m = []
	for i in r_to_c:
		r_m.append({v:k for k, v in i.items()})
	return(r_m)
	
	
# Below is a template Main which shows off some of the
# features of this library.
if __name__ == '__main__':

	# Grab and read a file.
	inputfile = input("Input File: ")
	inf = open(inputfile, 'r')
	cmap, rmaps, rmapscounts, nvoters = read_election_file(inf)
	
	# Pretty print to screen:
	pp_profile_toscreen(cmap, rmaps, rmapscounts)
	
	# Make a Borda scoring vector and evaluate the result.
	m = len(cmap)
	svec = [m - i for i in range(1,m+1)]
	scores = evaluate_scoring_rule(cmap, rmaps, rmapscounts, svec)
	
	#Pretty print results
	pp_result_toscreen(cmap, scores)