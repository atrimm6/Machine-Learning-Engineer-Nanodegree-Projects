def count_words(s,n):
	list_of_words = s.split()
	result=dict()
	for c in list_of_words:
		if c not in result:
			result[c] = 1
		else:
			result[c] += 1
	result = result.items()
	for i in range(len(result)):
		for j in range(len(result)-1-i):
			if result[j+1][1] > result[j][1]:
				result[j], result[j+1] = result[j+1], result[j]  
	for i in range(len(result)):
		for j in range(len(result)-1-i):
			if result[j+1][1] == result[j][1]:
				if result[j][0] > result[j+1][0]:
					result[j], result[j+1] = result[j+1], result[j] 
	top_n = result[:3]
	return top_n


def test_run():
    """Test count_words() with some inputs."""
    print count_words("cat bat mat cat bat cat",3)
    print count_words("betty bought a bit of butter but the butter was bitter",3)


if __name__ == '__main__':
    test_run()
