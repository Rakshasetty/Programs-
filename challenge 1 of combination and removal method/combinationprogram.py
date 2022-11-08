def comb(string, i=0):

    if i == len(string):   	 
        print("".join(string))

    for j in range(i, len(string)):

        res = [c for c in string]
   
        res[i], res[j] = res[j], res[i]
   	 
        comb(res, i + 1)

print(comb('123'))
