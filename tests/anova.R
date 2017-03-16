library(ez)

d <- data.frame(id=1:30, group=factor(PlantGrowth$group), weight=PlantGrowth$weight)
ezANOVA(d, weight, id, between=group, detailed=T)

d <- data.frame(id=1:60, supp=ToothGrowth$supp, dose=factor(ToothGrowth$dose), len=ToothGrowth$len)
ezANOVA(d, len, id, between=.(supp, dose), detailed=T)

d <- read.csv("toy.csv")
ezANOVA(d, v, id, between=.(a, b, c), detailed=T)

d(ANT)
ezANOVA(ANT, rt, subnum, between=group, detailed=T)

d <- read.csv("stress.csv")
ezANOVA(d, red, id, between=.(treat, age), detailed=T)

d <- read.csv("bushtucker.csv")
ezANOVA(d, value, participant, within=variable, detailed=T)

d <- read.csv("dating.csv")
ezANOVA(d, rating, id, between=gender, detailed=T)
ezANOVA(d, rating, id, within=pers, detailed=T)
