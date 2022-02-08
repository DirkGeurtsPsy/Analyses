###### Joran Lokkerbol; data exploration - sept 2019
###### Adapted by: Dirk Geurts, Gijs van de Veen, Fiona Zegwaard - nov 2020
options(scipen=999)
set.seed(12345)

# install.packages("rmarkdown")
# install.packages("psych")
# install.packages("ggplot2")
# install.packages("summarytools")
# install.packages("corrplot")
# install.packages("purrr")
# install.packages("tidyr")
# install.packages("caret")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("rattle")
# install.packages("dplyr")
# install.packages("DescTools")
# install.packages("PerformanceAnalytics")
# install.packages("caret")
# 
# # Necessary for packages to work in DRE
# install.packages("tmvnsim", dependencies=TRUE)
# install.packages("bitops", dependencies=TRUE)
# install.packages("gld", dependencies=TRUE)
# install.packages("Exact", dependencies=TRUE)
# install.packages("rootSolve", dependencies=TRUE)
 
library(rmarkdown)
        library(tmvnsim)
library(psych)
library(ggplot2)
library(corrplot)
library(purrr)
library(tidyr)
library(caret)
library(rpart)
library(rpart.plot)
        library(bitops)
library(rattle)
library(dplyr)
library(gld)
        library(Exact)
        library(rootSolve)
library(DescTools)
library(PerformanceAnalytics)
library(Caret)
rm(list = ls())  # Delete everything that is in R's memory
setwd("Z:/inbox/WorkInProgress") #set your working directory
origdataset <- read.csv("./Werkdatabestand/PredictDeprAI4Health.csv", header = TRUE)
origdataset <- as.data.frame(origdataset)
class(origdataset)


# Treatment effect Graph 1
origdataset$Colour = 'Grey'
origdataset$Colour[origdataset$Bdi2Sco_post >= (1.14*origdataset$Bdi2Sco_pre)] = 'Red'
# origdataset$Colour[origdataset$Bdi2Sco_post <= (0.86*origdataset$Bdi2Sco_pre)] = 'Green'

plot(x = origdataset$Bdi2Sco_pre, y = origdataset$Bdi2Sco_post, main = 'Treatment effect',
     xlab = 'Pre-treatment BDI score', ylab = 'Post-treatment BDI score',
     col = origdataset$Colour)


# Treatment effect Graph 2
origdataset$Treatment_effect = 'Not Deteriorated / Improved'
origdataset$Treatment_effect[origdataset$Bdi2Sco_post >= (1.14*origdataset$Bdi2Sco_pre)] = 'Deteriorated'
#origdataset$Treatment_effect[origdataset$Bdi2Sco_post <= (0.86*origdataset$Bdi2Sco_pre)] = 'Improved'

origdataset %>%
        ggplot(aes(x = origdataset$Bdi2Sco_pre, y = origdataset$Bdi2Sco_post, color = origdataset$Treatment_effect)) +
        geom_point() +
        labs(title = "Treatment effect of Mindfulness Based Cognitive Therapy",
             subtitle = "Difference in BDI score",
             x = "Pre Treatment score",
             y = "Post Treatment score",
             color = "Treatment effect")
       

## Descriptive statistics
#This file contains:
#* Exploration of outcome Y
#* Summary statistics 
#* Identification of near-zero variance predictors
#* Overview of missing data
#* Correlation matrix
#* Histograms of numerical variables
#* Boxplots of categorical variables with Y 
#* A simple regression tree including all predictors in the dataset
#* A simple regression tree including each predictor separately
#* A list of 'other things to look into'
#* An overview of the Results that can feed into next steps of the machine learning project 

#View(dataset)
str(origdataset)

## Organizing the dataset 
#  Remove redundant variables
# Toelichting wordt verwijderd, is alleen tekst van coder.
# Duplicate zijn er niet unique(origdataset$Duplicate)
# leeftijdscategory is redundent with respect to age
# Diffinvuldat is redundant with respect to dates of questionnaires; same goes for daysBtwnQ
# var1_pre is the same as ident
# Form_id just refers to number on quest set.
dataset <- origdataset %>% select(-c(Toelichting, Duplicate, LeeftCat, DiffInvulDat, daysBtwnQ, Var1_pre, Form_Id_pre))

# str(dataset$Sex)
# as.factor(dataset$Sex)

# changedatavectors to date format
dataset <-dataset %>% 
        mutate_at(.vars = vars(contains('Dat')&!contains('Diff')&!contains("Full")),as.Date, format ="%m/%d/%Y")

dataset<-dataset %>% rename(ID = ï..IdenNumm)

# Recode dashes to NA, yes no to 0, yes to 1
# for StopIntervention and Sex

dataset <- dataset %>% mutate_if(is.character, 
                                 ~recode(.,"-"="NA","no"="0","No"="0","Yes"="1","yes"="1"," 0"="0",
                                         "v"="1", "V"="1","M"="0","m"="0",
                                         "999"="NA", "888"="NA","o"="0"," "="NA","6"="6",
                                         "61-70"="7", "51-60" ="6", "41-50"="5", "31-40"="4","71-80"="8","0"="NA",
                                         "Lithium"="1",
                                         "g"= "NA","GG"="NA","Ongehuwd/ nooit gehuwd g"="NA"))
dataset$AlcoholAb<-recode(dataset$AlcoholAb,"6"="NA")

# change all data (except dates) to numeric; from there we can do easy recoding and making those variables that should be factors factors etc. 
dataset <- dataset%>%mutate_if(is.character, funs(as.integer(as.character(.))))

# change 999 to NA
dataset <- dataset %>% mutate_if(is.integer, 
                                 ~na_if(.,999))
dataset <- dataset %>% mutate_if(is.numeric, 
                                 ~na_if(.,999))
dataset <- dataset %>% mutate(Smoke, na_if(.,12))

# prepare input vectors


# Questionnaires
# We should first assess NA's on individudal items and imputate (modus? or median?)
# map the number of missings
datana <- sapply(dataset, function(x) sum(is.na(x))) 
str(datana)

# we coudl compare subscales of the literature later with tree-based partitioning based on our own data.

# Then make subscales. Subscales might be preferred, because they already contain the total scores by linear combination whereas you will lose information when you only include total scores. Cost of this reasoning is amount of predictors. Can be evaluated when running the models.
## BDI 3 subscales
# Cognitief: 3, 5,8,7,14,6,13
# Somatisch: 20,17,15,16,19,18,11,10,21
# Affectief: 2,9,12,4,1
dataset <- dataset%>%mutate(Q.BDI.cognitive = (Bdi203_pre+Bdi205_pre+Bdi207_pre+Bdi208_pre+Bdi214_pre+Bdi206_pre+Bdi213_pre))
dataset <- dataset%>%mutate(Q.BDI.somatic = (Bdi220_pre+Bdi217_pre+Bdi215_pre+Bdi216_pre+Bdi219_pre+Bdi218_pre+Bdi211_pre+Bdi210_pre+Bdi221_pre))
dataset <- dataset%>%mutate(Q.BDI.affective = (Bdi202_pre+Bdi209_pre+Bdi212_pre+Bdi204_pre+Bdi201_pre))
dataset %>% select(Q.BDI.cognitive, Q.BDI.somatic, Q.BDI.affective, ABdiScoIndIt, APswqScoIndIt, AOq45ScoIndIt, FfmqSco_pre, ZcvSco_pre)%>%
        chart.Correlation(.,histogram=TRUE, method = 'pearson', pch=19)

# total score pre and post
dataset <- dataset%>%mutate(Q.BDI.total.pre = ( Bdi201_pre+Bdi202_pre+Bdi203_pre+Bdi204_pre+Bdi205_pre+Bdi206_pre+Bdi207_pre+Bdi208_pre+Bdi209_pre+Bdi210_pre+
                                                Bdi211_pre+Bdi212_pre+Bdi213_pre+Bdi214_pre+Bdi215_pre+Bdi216_pre+Bdi217_pre+Bdi218_pre+Bdi219_pre+Bdi220_pre+
                                                Bdi221_pre ))

dataset <- dataset%>%mutate(Qout.BDI.total.post = ( Bdi201_post+Bdi202_post+Bdi203_post+Bdi204_post+Bdi205_post+Bdi206_post+Bdi207_post+Bdi208_post+Bdi209_post+Bdi210_post+
                                                        Bdi211_post+Bdi212_post+Bdi213_post+Bdi214_post+Bdi215_post+Bdi216_post+Bdi217_post+Bdi218_post+Bdi219_post+Bdi220_post+
                                                        Bdi221_post ))

dataset$Y <- dataset$Q.BDI.total.pre - dataset$Qout.BDI.total.post
# outcome y post - pre

## FFMQ 5 subscales
# Observeren = v6 + v10 + v15 + v20
# Beschrijven = v1 + v2 + v5 + v11 + v16
# Bewust handelen = v8 + v12 + v17 + v22 + v23
# Niet-oordelen = v4 + v7 + v14 + v19 + v24
# Non-reactief zijn = v3 + v9 + v13 + v18 + v21

#CHECK met:
# *1. Subschaal Observing (Obse) "Observeren";
# ARRAY ArrFfmObs(4) Ffmq06 Ffmq10 Ffmq15 Ffmq20;
# *2. Subschaal Describing (Desc) "Beschrijven";
# ARRAY ArrFfmDes(5) Ffmq01 Ffmq02 Ffmq05 Ffmq11 Ffmq16;
# *4. Subschaal Nonjudging of inner experience (Judg) "Niet-oordelen";
# ARRAY ArrFfmJud(5) Ffmq04 Ffmq07 Ffmq14 Ffmq19 Ffmq24;
# *3. Subschaal Acting with awareness (Acti) "Bewust handelen";
# ARRAY ArrFfmAct(5) Ffmq08 Ffmq12 Ffmq17 Ffmq22 Ffmq23;
# *5. Subschaal Nonreactivity to inner experience (Reac) "non-reactief zijn";
# ARRAY ArrFfmRea(5) Ffmq03 Ffmq09 Ffmq13 Ffmq18 Ffmq21;

dataset <- dataset%>%mutate(Q.FFMQ.observe = (dataset$Ffmq06_pre+dataset$Ffmq10_pre+dataset$Ffmq15_pre+dataset$Ffmq20_pre))
dataset <- dataset%>%mutate(Q.FFMQ.describe = (dataset$Ffmq01_pre+dataset$Ffmq02_pre+dataset$Ffmq05_pre+dataset$Ffmq11_pre+dataset$Ffmq16_pre))
dataset <- dataset%>%mutate(Q.FFMQ.actawareness = (dataset$Ffmq08_pre+dataset$Ffmq12_pre+dataset$Ffmq17_pre+dataset$Ffmq22_pre+dataset$Ffmq23_pre))
dataset <- dataset%>%mutate(Q.FFMQ.nonjudging = (dataset$Ffmq04_pre+dataset$Ffmq07_pre+dataset$Ffmq14_pre+dataset$Ffmq19_pre+dataset$Ffmq24_pre))
dataset <- dataset%>%mutate(Q.FFMQ.nonreactivity = (dataset$Ffmq03_pre+dataset$Ffmq09_pre+dataset$Ffmq13_pre+dataset$Ffmq18_pre+dataset$Ffmq21_pre))
dataset %>% select(Q.FFMQ.observe,Q.FFMQ.describe,Q.FFMQ.actawareness,Q.FFMQ.nonjudging,Q.FFMQ.nonreactivity)%>%
        chart.Correlation(.,histogram=TRUE, method = 'pearson', pch=19)

## Selfcompassion 6 subscales
# Subschalen Zelfcompassie vragenlijst NL from litarature:
# +Self kindness items:  5, 7, 11, 13 
# -Self Judgment items:  16, 20, 21, 23, 
# +Common Humanity items: 3, 4, 12, 19
# -Isolation items: 2, 9, 15, 17
# +Mindfulness Items: 8, 14, 22, 24
# -Over-identification items: 1, 6, 10, 18

# CHECKED WITH:
# *ZCV1.Subschaal Selfkindness (Ski)";
#  ARRAY ZcvSkiArr(4) Zelf05 Zelf07 Zelf11 Zelf13;
# *ZCV2.Subschaal Selfjudgement (Sju)";
# ARRAY ZcvSjuArr(4) Zelf16 Zelf20 Zelf21 Zelf23;
# *ZCV3.Subschaal CommHumm (Com)";
#  ARRAY ZcvComArr(4) Zelf03 Zelf04 Zelf12 Zelf19;
# *ZCV4.Subschaal Isol (Iso)";
# ARRAY ZcvIsoArr(4) Zelf02 Zelf09 Zelf15 Zelf17;
# *ZCV5.Subschaal Mindfulness (Min)";
#  ARRAY ZcvMinArr(4) Zelf08 Zelf14 Zelf22 Zelf24;
#  *ZCV6.Subschaal Overid (Ove) (wat is dat?)";
# ARRAY ZcvOveArr(4) Zelf01 Zelf06 Zelf10 Zelf18;

dataset <- dataset%>%mutate(Q.SCS.selfkindness =(Zelf05_pre+Zelf07_pre+Zelf11_pre+Zelf13_pre))
dataset <- dataset%>%mutate(Q.SCS.selfjudgement =(Zelf16_pre+Zelf20_pre+Zelf21_pre+Zelf23_pre))
dataset <- dataset%>%mutate(Q.SCS.commonhumanity =(Zelf03_pre+Zelf04_pre+Zelf13_pre+Zelf19_pre))
dataset <- dataset%>%mutate(Q.SCS.isolation =(Zelf03_pre+Zelf09_pre+Zelf15_pre+Zelf17_pre))
dataset <- dataset%>%mutate(Q.SCS.mindfulness =(Zelf08_pre+Zelf14_pre+Zelf24_pre+Zelf22_pre))
dataset <- dataset%>%mutate(Q.SCS.overidentified =(Zelf01_pre+Zelf06_pre+Zelf10_pre+Zelf18_pre))

## PSWQ 2 subscales 
# CHECK datamanager code:
# *Omkeer- of reverse-items omdraaien ("terugspiegelen"): hogere score => hogere mate van worry;
# ARRAY PswqOmk{5} Pswq01 Pswq03 Pswq08 Pswq10 Pswq11;

# Worry: items 2,4,5,6,7,9,12,13,14,15,16
# Absence of worry: items 1, 3, 8, 10, 11
dataset <- dataset%>%mutate(Q.PSWQ.worry = (Pswq02_pre+Pswq04_pre+Pswq05_pre+Pswq06_pre+Pswq07_pre+Pswq09_pre+Pswq12_pre+Pswq13_pre+Pswq14_pre+Pswq15_pre+Pswq16_pre))
dataset <- dataset%>%mutate(Q.PSWQ.AbsenceOfWorry = (Pswq01_pre+Pswq03_pre+Pswq08_pre+Pswq10_pre+Pswq11_pre))



# OQ --> leave out (might see how well it correlates with other measures)
# *Subschaal SD: Symptom Distress (Subschaal (ernst van de) symptomen);
# ARRAY Oq45Sd{25} Oq4502 Oq4503 Oq4505 Oq4506 Oq4508 Oq4509 Oq4510 Oq4511 Oq4513 Oq4515 Oq4522 Oq4523
# Oq4524 Oq4525 Oq4527 Oq4529 Oq4531 Oq4533 Oq4534 Oq4535 Oq4536 Oq4540 Oq4541 Oq4542 Oq4545;
# *Subschaal SR: Social Role (Sociale Rol of Maatschappelijk Functioneren);
# ARRAY Oq45Sr{9} Oq4504 Oq4512 Oq4514 Oq4521 Oq4528 Oq4532 Oq4538 Oq4539 Oq4544;
# ****Subschaal IR: Interpersonal Relations (Interpersoonlijk Functioneren);
# ARRAY Oq45Ir{11} Oq4501 Oq4507 Oq4516 Oq4517 Oq4518 Oq4519 Oq4520 Oq4526 Oq4530 Oq4537 Oq4543;

dataset <- dataset%>%mutate(Q.OQ.symptomdistress =(Oq4502_pre + Oq4503_pre + Oq4505_pre + Oq4506_pre + Oq4508_pre + Oq4509_pre + Oq4510_pre + Oq4511_pre + Oq4513_pre + Oq4515_pre + Oq4522_pre + Oq4523_pre))
dataset <- dataset%>%mutate(Q.OQ.socialrole =(Oq4504_pre + Oq4512_pre + Oq4514_pre + Oq4521_pre + Oq4528_pre + Oq4532_pre + Oq4538_pre + Oq4539_pre + Oq4544_pre))
dataset <- dataset%>%mutate(Q.OQ.interpersonalrelations =(Oq4501_pre + Oq4507_pre + Oq4516_pre + Oq4517_pre + Oq4518_pre + Oq4519_pre + Oq4520_pre + Oq4526_pre + Oq4530_pre + Oq4537_pre + Oq4543_pre))

# demographic variables (dummy coding?)
# Eduation level (3 levels?)
dataset <- dataset %>% mutate(Dem.education = case_when(
                                        Opleiding<4 ~ 1,
                                        Opleiding>3 & Opleiding<6 ~ 2,
                                        Opleiding>5 ~ 3,
                                        is.na(Opleiding) ~ 0))
dataset$Dem.education.lower <-          dataset$Dem.education==1
dataset$Dem.education.medium <-         dataset$Dem.education==2
dataset$Dem.education.higher <-         dataset$Dem.education==3
dataset$Dem.education.missing <-        dataset$Dem.education==0

#Age
dataset$Dem.age<-dataset$AgeStart

#Gender
dataset$Dem.gender <- dataset$Sex

#dat$Edulevel    = factor(dat$Edulevel, levels=c(1,2,3), labels=c("Lower","Intermediate","Higher"))

# # Work status (4 levels?)
# Werk	0	missing
# 1	Betaald werk
# 10	volgt onderwijs/ studeert
# 2	In ziektewet
# 5	(vervroegd) met pensioen (AOW, VUT, FPU)
# 6	Werkeloos/ werkzoekend (geregistreerd bij arbeidbureau)
# 7	Arbeidsongeschikt (WAO, AAW, WAZ, WAJONG, WIA)
# 8	Bijstandsuitkering
# 9	Fulltime huisvrouw/ huisman
# NB no 3 or 4 coded

#         
#  Dicussion what to bin together to get sufficient groups: 
# 1) payed work, 
# 2) student/housewife/retired (in combination with age more or less retrievable by algorithm), 
# 3) sick leave 
# 4) not working
dataset <- dataset %>% mutate(Dem.work = case_when(
        is.na(Werk)~0,
        Werk==0~0,
        Werk==1~1,
        (Werk==10|Werk==9|Werk==5)~2,
        Werk==2~3,
        (Werk==8|Werk==6|Werk==7|Werk==8)~4))
dataset$Dem.work.payed <- dataset$Dem.work==1
dataset$Dem.work.nonpayed <- dataset$Dem.work==2
dataset$Dem.work.sickleave <- dataset$Dem.work==3
dataset$Dem.work.nowork <- dataset$Dem.work==4
dataset$Dem.work.missing <- dataset$Dem.work==0

# Marital status
Freq(factor(dataset$BurgeStaat))
dataset <- dataset %>% mutate(Dem.maritalstatus = case_when(
        is.na(BurgeStaat)~0,
        BurgeStaat==0~0,
        BurgeStaat==1~1,
        BurgeStaat==2~2,
        BurgeStaat==3~3,
        (BurgeStaat==4|BurgeStaat==5)~4,
        BurgeStaat==6~0))
dataset$Dem.matitalstatus.married <- dataset$Dem.work==1
dataset$Dem.maritalstatus.livingtogether <- dataset$Dem.work==2
dataset$Dem.maritalstatus.nonmarried <- dataset$Dem.work==3
dataset$Dem.maritalstatus.divorcedwidowed <- dataset$Dem.work==4
dataset$Dem.maritalstatus.missing <- dataset$Dem.work==0
# clinical information
## Treatment history intensity (no, inpatient 17%, daytreatment 10%, outpatient)
# take existing variables
dataset <- dataset %>%rename_with(.,.fn = ~paste0('Clin.', .), .cols = starts_with('Prev'))

# Treatment history psychoterapy
# Take existing variables

# Treatment history medication
# Take existing variables

## Current Medication: divide by group (serotonergic, dopaminergic, benzodiazepine)
# MedDep1	0	No use
# 1	MAO-inhibitor
# 2	TCA
# 3	SSRI
# 4	SNRI
# 5	Other
# MedDep2	0	No use
# 1	TCA
# 2	Lithium addition
# 3	SSRI
# 4	SNRI
# 5	Other
# MedBip	0	No use
# 1	Lithium
# 2	Lamotrigine
# 3	Olanzapine
# 4	Lamotrigine+Olanzapine
# 5	Lithium+Anti-epilepticum
# MedBenzo	0	No use
# 1	Use
# MedAntipsych	0	No use
# 1	Typical antipsychotic
# 2	Atypical antipsychotic
# MedSleep	0	No use
# 1	Melatonin
# 2	Zoplicon
# 3	Zolpidem

MedicationTable <- dataset %>% select(contains('Med')) %>% gather()%>% table(.)

#Benzo as is
dataset <- dataset%>%mutate(Clin.Med.Benzo = case_when(
        is.na(MedBenzo)~0,
        MedBenzo==1~1,
        MedBenzo==0~0,
        MedSleep==2|MedSleep==3~1))

# Antipsychotics, combine typical and atypical
dataset <- dataset%>%mutate(Clin.Med.AP = case_when(
        is.na(MedAntipsych)~0,
        MedAntipsych==1~1,
        MedAntipsych==2~1,
        MedAntipsych==0~0,
        MedBip==4~1))

# lithium: add addition and bipolar
dataset <- dataset%>%mutate(Clin.Med.Lithium = case_when(
        MedBip==1~1,
        MedBip==5~1,
        MedDep2==2~1))

# SSRI: add combis too
dataset <- dataset%>%mutate(Clin.Med.SRI = case_when(
        MedDep1==3|MedDep1==4~1,
        MedDep2==3|MedDep2==4~1))

# TCA: add combis too
dataset <- dataset%>%mutate(Clin.Med.TCA = case_when(
        MedDep1==2~1,
        MedDep2==1~1))

# Other: MAO (6), other (28), lamotrigine (4), anti-epilepticum (3), melatonin(25) 
dataset <- dataset%>%mutate(Clin.Med.Other = case_when(
        MedDep1==1 | MedDep1==5~1,
        MedDep2==5~1,
        MedBip==2 | MedDep1==4|MedDep1==5~1,
        MedSleep == 1~1))

## Drugs
# Smoke
dataset <- dataset%>%mutate(Clin.Intox.smoke = case_when(
        is.na(Smoke)|Smoke==0~0,
        Smoke==1 | Smoke==12~1))

# Alcoholuse
dataset$Clin.Intox.Alcohol <- dataset$Alcohol

# General Druguse ONLY 43 persons.

## Diagnostic categories
# Presence of current depression/dysthymia
# Make vector with Current/not Current depression
dataset <- dataset%>%mutate(DSM.CurrentDepres = case_when(
        DSMDepSt==1|DSMDepSt==4 ~1,
        DSMDysthym==1 ~1,
        DSMDepSt==2|DSMDepSt==3|DSMDepSt==5|DSMDepSt==6 ~0))

# Make vector with recidivicing non-recidivicing
dataset <- dataset%>%mutate(DSM.RecurrentDepres = case_when(
        DSMDepSt==1|DSMDepSt==2|DSMDepSt==3 ~0,
        DSMDepSt==4|DSMDepSt==5|DSMDepSt==6 ~1))

# partial remission
dataset <- dataset%>%mutate(DSM.PartlyRemittedDepres = case_when(
        DSMDepSt==2|DSMDepSt==5 ~1,
        DSMDepSt==1|DSMDepSt==3|DSMDepSt==4|DSMDepSt==6~0))
# Not necessary recidivicing without current = remitted

# Presence of ADHD
dataset <- dataset%>%mutate(DSM.ADHD = case_when(
        DSMOntwikk==0|DSMOntwikk==1 ~0,
        DSMOntwikk==2|DSMOntwikk==3 ~1))

# Presence of ASD
dataset <- dataset%>%mutate(DSM.ASD = case_when(
        DSMOntwikk==0|DSMOntwikk==2 ~0,
        DSMOntwikk==1|DSMOntwikk==3 ~1))


# Presence of somatoform disorder
dataset <- dataset%>%mutate(DSM.SomaticSymptoms = case_when(
        DSMSomaSt==1 ~ 1,
        DSMSomaSt==0|DSMSomaSt==2|DSMSomaSt==3 ~ 0))

# Presence of Panic disorder
dataset <- dataset%>%mutate(DSM.PanicDisorder = case_when(
        DSMAngst1==1|DSMAngst1==2|DSMAngst2==1|DSMAngst2==2 ~ 1,
        DSMAngst1!=1&DSMAngst1!=2&DSMAngst2!=1&DSMAngst2!=2 ~ 0))

# Presence of Social anxiety
dataset <- dataset%>%mutate(DSM.SocialAnxiety = case_when(
        DSMAngst1==3|DSMAngst1==4 ~ 1,
        DSMAngst1!=3&DSMAngst1!=4 ~ 0))

# Presence of General anxiety
dataset <- dataset%>%mutate(DSM.GeneralAnxiety = case_when(
        DSMAngst1==6|DSMAngst2==3 ~ 1,
        DSMAngst1!=6&DSMAngst1!=3 ~ 0))

# Other Anxiety
dataset <- dataset%>%mutate(DSM.AnxietyOther = case_when(
        DSMAngst1==4|DSMAngst1==5|DSMAngst1==7|DSMAngst1==8 ~ 1,
        DSMAngst2==5|DSMAngst2==6|DSMAngst2==7|DSMAngst2==8 ~ 1,
        DSMSoma == 6 ~1,
        DSMAngst1!=4&DSMAngst1!=5&DSMAngst1!=7&DSMAngst1!=8 ~ 0,
        DSMAngst2!=5&DSMAngst2!=6&DSMAngst2!=7&DSMAngst2!=8 ~ 0,
        DSMSoma != 6 ~0))

# Anxiety disorder status
dataset <- dataset%>%mutate(DSM.AnxietyOther = case_when(
        DSMAngstSt==2 ~ 1,
        DSMAngstSt==1 ~ 2))

# Presence of personality problems, traits/full disorder
dataset <- dataset%>%mutate(DSM.PersonalityDisOrder = case_when(
        DSMAs2==1|DSMAs2==2|DSMAs2==3|DSMAs2==4|DSMAs2==5|DSMAs2==6|DSMAs2==7|DSMAs2==8|DSMAs2==9|DSMAs2==10|DSMAs2==11 ~ 1))

dataset <- dataset%>%mutate(DSM.PersonalityTraits = case_when(
        DSMAs2==13|DSMAs2==14|DSMAs2==15|DSMAs2==16|DSMAs2==17|DSMAs2==18|DSMAs2==19|DSMAs2==20|DSMAs2==21|DSMAs2==22 ~ 1))

# Presence of other diagnosis
dataset <- dataset%>%mutate(DSM.Other = case_when(
        DSMEetSt==1| DSMEetSt==2| DSMEetSt==3 ~ 1,
        DSMVerslaving==1 ~ 1,
        DSMCognitief==1| DSMCognitief==2| DSMCognitief==3| DSMCognitief==4 ~ 1,
        DSMRestSt == 1 ~1,
        DSMBipolair==1|DSMBipolair==2 ~1))


# Presence of somatic problems (very iffy vector)
# Think of how to code (depending on group sizes: ordinal 0, 1, 2 or more than 2)
SomSummary <- dataset %>% mutate_at(vars(starts_with("Som")), funs(case_when(.>0 ~ 1, .==0 ~0)))%>%
                        select(starts_with("Som"))%>%
                        mutate(Clin.Somatic=rowSums(.,na.rm=TRUE))%>%
                        mutate(Clin.Somatic, Clin.SomaticCat = case_when(Clin.Somatic>2~3,
                                                       Clin.Somatic==2~2,
                                                       Clin.Somatic==1~1,
                                                       Clin.Somatic==0~0))
#dataset <- cbind.data.frame(dataset,SomSummary$Clin.SomaticCat)                                                
dataset$Clin.SomaticCat <-SomSummary$Clin.SomaticCat
# change all NAs in DSM. and Clin. in 0.
# dataset <- dataset %>% mutate_at(vars(starts_with("DSM.")|starts_with("Clin.")),~replace(.,is.na(.),0))

### Create dataset to predict drop out ###
datasetDO <- dataset
datasetDO$Y <- datasetDO$Numbses<5

set.seed(1)
trainDO <- createDataPartition(datasetDO$Y,p = 0.75, list = FALSE)
df_trainDO <- datasetDO[trainDO,]
df_testDO <- datasetDO[-trainDO,]

# save split 
write.csv(df_testDO,"Z:/inbox/TestSet/df_testDO")
write.csv(df_trainDO,"Z:/inbox/WorkInProgress/Werkdatabestand/df_trainDO")

### EXCLUSION ###

# Now exclude those patients with only 1 day assessment, (remove those without evaluation date), as a consequence there is no BDI to predict for them.
dataset <- dataset %>% filter(!is.na(Y))

# If more than half (>7) of questionaire subscales (except OQ) is missing remove:
# Q.VarMissings <- dataset %>% select(starts_with("Q.")& -starts_with("Q.OQ")) %>% summarise_all(funs(sum(is.na(.))))
# Q.IDMissings <- dataset %>% select(starts_with("Q.")& -starts_with("Q.OQ")) %>% mutate(NAQ = rowSums(is.na(.)))
# dataset <- cbind(dataset,Q.IDMissings$NAQ)
# exclusion <- dataset %>% filter(Q.IDMissings$NAQ>7)
# dataset <- dataset %>% filter(Q.IDMissings$NAQ<8)


# Construct input data.frame DSM. Clin. Dem.
# in toto

# split dataset into training 3/4 and test dataset  1/4
set.seed(1)
train <- createDataPartition(dataset$Y,p = 0.75, list = FALSE)
df_train <- dataset[train,]
df_test <- dataset[-train,]

# save split 
write.csv(df_test,"Z:/inbox/TestSet/df_test")
write.csv(df_train,"Z:/inbox/WorkInProgress/Werkdatabestand/df_train")


# Process training dataset further 
# Impute data
        # for demographics add NA group
# No NAs anymore at this point (group for missing)

        # for questionnaires impute rounded median
df_train %>% select(starts_with("Q.")) %>% summarise_all(funs(sum(is.na(.))))

# df_train <- df_train %>% transmute_if(is.integer, funs(as.numeric(as.integer(.))))
str(df_train)
medians.Q <- df_train%>% select(starts_with("Q.")) %>% summarise(across(everything(),~ median(., na.rm =TRUE)))

df_train <- df_train %>% mutate_at(vars(starts_with("Q.")), ~ifelse(is.na(.), median(., na.rm =TRUE,), .))
df_train %>% select(starts_with("Q.")) %>% summarise_all(funs(sum(is.na(.))))
                                                              
        # for DSM imputation NA --> no / 0 
df_train %>% select(starts_with("DSM.")) %>% summarise_all(funs(sum(is.na(.))))
df_train <- df_train %>% mutate_at(vars(starts_with("DSM.")), ~ifelse(is.na(.),0,.))

# for Clin
# df_train %>% select(starts_with("Clin")) %>% str()
df_train %>% select(starts_with("Clin.")) %>% summarise_all(funs(sum(is.na(.))))
df_train <- df_train %>% mutate_at(vars(starts_with("Clin."),-starts_with("Clin.Intox.Alcohol")), ~ifelse(is.na(.),0,.))
medians.alcohol <- median(df_train$Clin.Intox.Alcohol)
df_train <- df_train %>% mutate_at(vars(starts_with("Clin.Intox.Alcohol")), ~ifelse(is.na(.), median(., na.rm =TRUE,), .))

# Select predictors
Input <- df_train%>%select((starts_with("DSM.")|starts_with("Clin.")|starts_with("Dem.")|starts_with("Q."))&-starts_with("ID"))

# Check near zero variance
## Identify near-zero variances predictors
nearZeroVar(Input, saveMetrics = TRUE)

# Check for outliers with correlation matrix for numeric predictors
Input %>% select(Q.OQ.symptomdistress, Q.OQ.socialrole, Q.OQ.interpersonalrelations,
                   Q.BDI.cognitive, Q.BDI.somatic, Q.BDI.affective,
                   Q.PSWQ.worry, Q.PSWQ.AbsenceOfWorry,
                   Q.FFMQ.observe,Q.FFMQ.describe,Q.FFMQ.actawareness,Q.FFMQ.nonjudging,Q.FFMQ.nonreactivity,
                   Q.SCS.selfjudgement,Q.SCS.commonhumanity,Q.SCS.mindfulness,
                   Q.SCS.overidentified,Q.SCS.selfkindness,Q.SCS.isolation,
                   Dem.age, Clin.Intox.Alcohol)%>%
        chart.Correlation(.,histogram=TRUE, method = 'pearson', pch=19)


# Check whether factors are different for those who did and did not fill out second quest?

# Next make vectors that are indeed categorical factors
# Input <- Input%>%
#         mutate_at(vars(-starts_with('Q.')&
#                        -starts_with('Dem.age')&
#                        -starts_with('Clin.Intox.Alcohol')&
#                         -starts_with('Clin.SomaticCat')),
#                                funs(as.factor(.)))
# Input$Clin.SomaticCat <- as.ordered(Input$Clin.SomaticCat)
# str(Input)
## to be very nice to ourselves we might label all factors (but for now we leave it at this)
                     
## Describing the data (general)
Input %>% 
        keep(is.numeric) %>% describe

#suppressWarnings(print(dfSummary(dataset), method = 'render', silent=TRUE))


# map levels and frequencies per categorical variable
temp <-  Input %>% keep(is.factor)
for (i in 1:ncol(temp)) {
        print(names(temp[i]))
        print(Freq(temp[,i]))}

## Histograms per variable to check distribution and outliers
Input %>%
        select(contains('Q.'))%>%
        keep(is.numeric) %>%
        gather() %>%
        ggplot(aes(value)) +
        facet_wrap(~ key, scales = "free") +
        ylim(0,150)+
        geom_histogram(binwidth = 2)


## Exploring the outcome Y

##### describe Y #####
plot(df_train$Y)
table(df_train$Y)
summary(df_train$Y)
boxplot(df_train$Y)
histogram(df_train$Y)
plot.ecdf(df_train$Y)
sum(is.na(df_train$Y)) #nr of missings affect generalizability of your model


## Scatterplots to check for (non-)linear relations
temp <- Input %>% keep(is.numeric)
a <- ncol(temp)
b <- a/2
pairs(df_train$Y~.,data=temp[,c(1:b,a)], main="Scatterplot Matrix")
pairs(df_train$Y~.,data=temp[,c((b+1):a)], main="Scatterplot Matrix")



## Boxplots of categorical variables and Y 
bp <- dataset[unlist(lapply(dataset, is.factor))]
bp$Y <- as.numeric(df_train$Y)
for(k in 1:ncol(bp)){
        print(names(bp[k]))
        boxplot(bp$Y ~ bp[, k])
        #ggplot(bp, aes(x=bp[,k], y=bp$Y)) + 
        #  geom_boxplot()
}


## Chi-square tests of categorical variables with Y
cs <- Input[unlist(lapply(Input, is.factor))]
cs$Y <- df_train$Y
for(k in 1:(ncol(cs)-1)){
        print(names(cs[k]))
        print(chisq.test(cs$Y, cs[, k]), correct = FALSE)
        #ggplot(bp, aes(x=bp[,k], y=bp$Y)) + 
        #  geom_boxplot()
}

## Simple regression tree with all predictors and Y
tree <- rpart(df_train$Y~., data = Input)
rpart.plot(tree, type=0)

## Simple regression tree for all predictors separately and Y
## Rpart or every single x and y 
for(i in 1:ncol(Input)){
        print(names(Input)[i])
        tree <- rpart(df_train$Y~Input[,i], data = Input)
        rpart.plot(tree)   }

## Regression tree for a set of correlated predictors on Y
temp <- select(df_train, Y, G1, G2, G3)
tree <- rpart(df_train$Y~., data = temp)
rpart.plot(tree)

## Other things to look into
#* do infeasible combinations of variable values occur in the data (e.g. minors with a drivers license or pregnant males)? 
#* a tree-model where Y is being predicted using a cluster of related X-variables, such as ROM-items
#* which variables are known / not known at the point of prediction?
#* which domains (work, health, family, lifestyle, therapy, etc etc) are covered?


## Results
#* assessment of the quality of the data (in terms of outliers and missings)
#* input regarding the moment of prediction
#* input for data cleaning (handling missing data; removing variables not known at time of prediction, near-zero variance variables, etc)
#* input for feature engineering (adjusting variables based on tree-analyses, based on correlations, based on domain-analysis)
#* input for defining the outcome variable Y
#* input for defining the project in terms of generalizability (in case of missing Y values)
#* input for choosing the project in case there are still multiple options at the table
#* input for defining the scope of the project (e.g. limiting to a subgroup to get a better balanced outcome variable)
#* a potential revision of the goal of your project
#* input for which variables and combination of variables seem particularly relevant within the to-be-developed algorithms 

install.packages('ISLR')
install.packages('rpart')
install.packages('rpart.plot')

library(ISLR)
library(rpart)
library(rpart.plot)


### use 
fitControl <- trainControl(## 10-fold CV
        method = "repeatedcv",
        number = 10,
        ## repeated ten times
        repeats = 3)

trainData <- Input %>% select(-c(Dem.education,Dem.work,Dem.maritalstatus))

# preprocess data 
preProcess_range_model <- preProcess(trainData, method='range')
trainData <- predict(preProcess_range_model, newdata = trainData)
trainData$Y <- df_train$Y
apply(trainData[,1:10],2 , FUN=function(x){c('min'=min(x),'max'=max(x))})

# train model (GBM)
set.seed(825) # for reproducibility

# available algorithms
modelnames <- paste(names(getModelInfo()), collapse =', ')
modelnames

# Train the model using randomForest and predict on the training data itself
model_mars = train(Y~., data=trainData, method='rpart', metric="RMSE", tuneLength = 30, trControl = fitControl)
fitted <- predict(model_mars)
plot(model_mars, main = "Model Accuracies with MARS")

# compute variable importance
varimp_mars <- varImp(model_mars)
plot(varimp_mars)
