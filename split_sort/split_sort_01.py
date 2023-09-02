# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 11:42:03 2023

@author: cromlyn
"""

import csv # libary for reading and writing csv files
import random

def get_job_names(filename):
    list_of_jobs = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            list_of_jobs.append(row)
            line_count += 1
    print(f'Processed {line_count} lines.')
    return list_of_jobs

def save_short_list(jobs):
    filename = "jobs_"+str(len(jobs))+'csv'
    with open(filename, mode='w') as out_file:
        wr = csv.writer(out_file, quoting=csv.QUOTE_MINIMAL)
        wr.writerow(jobs)


# starting with rouglhy 400 jobs,
# looking to filter to top 15-20  <10%

# split pile to two and keep better half. 400,200,100,50,25,15ish = 790 comparisons
# split pile to four and keep better quarter 400, 100, 25, 15ish = 540 comparisons, better, but not wahat asked for

def split_list(jobs, downto):
    r = random.randint(0, len(jobs)) # choose a cutter randomly from list
    cutter = jobs[r]
    keep = []
    leave = []
    for i in range (0, len(jobs)):
        testJob = jobs[i]
    
        if i == r:
            print("Keep ", cutter, "?")
            key = input("y/n  ") 
        else:
            print("Prefer ", testJob, " over ", cutter, "?  ", i, "/",  len(jobs))
            key = input("y/n  ")

        if key == "y":
            keep.append(testJob)
        elif key == "n":
            leave.append(testJob)
        else:
            print("didn't recognise input key") # will not keep any job with a weird 
    if len(keep) > downto:
        print( len(keep), " jobs left to rank. saving and continuing")
        save_short_list(keep)
        keep = split_list(keep, downto)
    return(jobs)
            
            
        
filename = 'jobs.csv'
downto = 10 # go down to less then 20 left
job_list = get_job_names(filename)
job_list = split_list(job_list, downto)
print("_____________")
print("final short list to be considered is...") # can repalce this bit with bubble sort if really needed
print(job_list)


        