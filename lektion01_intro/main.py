#print("Hello World!")

# Indlæs fra bruger
number = int(input("Indast et heltal: "))

# Tjek om tallet er større end 3
if number > 3:
    print("Hello World!")

print("----")

forfattere = ["Hans", "Jens", "Peter", "Jørgen", "Børge"]
for forfatter in forfattere:
    print(forfatter)

print("----")
forfattere.append("Anders")
for forfatter in forfattere:
    print(forfatter)

print("----")

del forfattere[1]
print("Antal forfattere i listen: ", len(forfattere))

print("----")
forfattere.reverse()
for forfatter in forfattere:
    print(forfatter)
