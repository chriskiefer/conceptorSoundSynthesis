import numpy as np
from functools import reduce

def criterionMax(a,b):
    return a > b

def criterionMin(a,b):
    return a < b

def createPop(N, geneSize, scoreToBeat, binary=False):
    pop = {'genes': np.random.random((N, geneSize)),
           'bestIdx':-1, 'bestScore':scoreToBeat,
           'scores':np.zeros(N) + scoreToBeat,
           'scoreHist':[],
           'meanHist':[],
          'reevaluate':np.ones(N) > 0}
    if binary:
        pop['genes'] = pop['genes'] > 0.5
    return pop
 
def chooseTwo(popSize, deme):
    offset = np.random.randint(popSize)
    candidates = np.mod(np.arange(deme) + offset, popSize)
    np.random.shuffle(candidates)
    return candidates[:2]
    
def fight(competitors, evaluationFunc, evaluationCriterion, data, reeval, currentScores):
    #print(competitors)
    scores = [currentScores[x] if not reeval[x] else evaluationFunc(competitors[x], data) for x in np.arange(2)]
    winner = 0 if evaluationCriterion(scores[0], scores[1]) == True else 1
    loser = 1 - winner
    return [winner, loser, scores]

def infect(winner, loser, recombChance, mutateChance, mutateScale):
    mutateChance = min(mutateChance + np.random.power(1/8), 1.0)
#     recombChance = min(recombChance + np.random.power(1/8), 1.0)
    for i in range(winner.size):
        if(np.random.rand() < recombChance):
            loser[i] = winner[i]
    if (loser.dtype == np.bool):
        loser = np.array([x if np.random.rand() > mutateChance else not x for x in loser])
    else:
        loser = loser + (np.random.randn(loser.size) * (np.random.rand(loser.size) < mutateChance) * mutateScale)
        loser = np.clip(loser, 0, 1)
    return loser

def evolve(pop, evaluationFunc, evaluationCriterion, data, deme, epochs, recombChance=0.5, mutateChance=0.5, mutateScale = 1.0, onEpochFunc=None):
    for i_epoch in range(epochs):
        if (onEpochFunc):
            onEpochFunc()
        print("=============== epoch ", i_epoch)
        print("Scores: ", pop['scores'], " best: ", pop['bestScore'], pop['bestIdx'], " mean: ", pop['meanHist'][-1] if len(pop["meanHist"]) > 0 else "n/a")
        print(pop["genes"])
        
        competitors = chooseTwo(pop['genes'].shape[0], deme)
        winner, loser, scores = fight(pop['genes'][competitors], evaluationFunc, evaluationCriterion, data, 
                                      pop['reevaluate'][competitors], pop['scores'][competitors])
        print("\nfight: ", competitors, ", ", winner)
        #print (winner, loser, scores)
#        if evaluationCriterion(scores[winner], pop['bestScore']) == True:
#            pop['bestScore'] = scores[winner]
#            pop['bestIdx'] = competitors[winner]
        loserGene = infect(pop['genes'][competitors[winner]], pop['genes'][competitors[loser]], recombChance, mutateChance, np.var(pop['genes']) * mutateScale + 0.0001) 
#         pop['genes'][competitors[winner]] = winnerGene
        pop['genes'][competitors[loser]] = loserGene
        pop['scores'][competitors[winner]] = scores[winner]
        pop['scores'][competitors[loser]] = scores[loser]
        pop['reevaluate'][competitors[winner]] = False
        pop['reevaluate'][competitors[loser]] = True
        pop['bestScore'],pop['bestIdx'] = reduce(
            lambda p1,p2: p1 if evaluationCriterion(p1,p2) else p2, 
            ((x,i) for i,x in enumerate(pop['scores']))
        ) 
        popMean = np.mean(pop['scores'])
        pop['meanHist'].append(popMean)
        pop['scoreHist'].append(pop['bestScore'])

    return pop

