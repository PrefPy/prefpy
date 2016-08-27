import numpy as np
from scipy.stats import rankdata

def GenerateRUMParameters(m, distribution):
    arr = []
    for i in range(0, m):
        arr.append(1)
    if distribution=='normal':
        parameter = dict(m = m, Mean = np.random.uniform(0,1,(m,)), SD = np.array(arr))
        parameter["order"] = parameter["Mean"].ravel().argsort()
    elif distribution=='exponential':
        unscaled = random.uniform(0,1,(m,))
        parameter = dict(m = m, Mean = unscaled/unscaled.sum())
    else:
        raise ValueError("Distribution name \"", distribution, "\" not recognized")
    return parameter

#' Generate observation of ranks given parameters
#'
#' Given a list of parameters (generated via the Generate RUM Parameters function),
#' generate random utilities from these models and then return their ranks
#'
#' @param Params inference object from an Estimation function, or parameters object from a generate function
#' @param m number of alternatives
#' @param n number of agents
#' @param distribution can be either 'normal' or 'exponential'
#' @return a matrix of observed rankings
#' @export
#' @examples
#' Params = Generate.RUM.Parameters(10, "normal")
#' Generate.RUM.Data(Params,m=10,n=5,"normal")
#' Params = Generate.RUM.Parameters(10, "exponential")
#' Generate.RUM.Data(Params,m=10,n=5,"exponential")
def GenerateRUMData(Params, m, n, distribution):
    if distribution == "exponential":
        return np.transpose(np.repeat(rankdata(np.random.exponential(size = m, scale = 1/Params["Mean"])), n))
    elif distribution == "normal":
        A = rankdata(-np.random.normal(size = m, loc = Params["Mean"], scale = Params["SD"]))-1
        for i in range(0, n - 1):
            A = np.vstack([A, (-np.random.normal(size = m, loc = Params["Mean"], scale = Params["SD"])).ravel().argsort()])
        return A.astype(int)
    else:
        raise ValueError("Distribution name \"", distribution, "\" not recognized")#' Breaks full or partial orderings into pairwise comparisons
#'
#' Given full or partial orderings, this function will generate pairwise comparison
#' Options
#' 1. full - All available pairwise comparisons. This is used for partial
#' rank data where the ranked objects are a random subset of all objects
#' 2. adjacent - Only adjacent pairwise breakings
#' 3. top - also takes in k, will break within top k
#' and will also generate pairwise comparisons comparing the
#' top k with the rest of the data
#' 4. top.partial - This is used for partial rank data where the ranked
#' alternatives are preferred over the non-ranked alternatives
#'
#' The first column is the preferred alternative, and the second column is the
#' less preferred alternative. The third column gives the rank distance between
#' the two alternatives, and the fourth column enumerates the agent that the
#' pairwise comparison came from.
#'
#' @param Data data in either full or partial ranking format
#' @param method - can be full, adjacent, top or top.partial
#' @param k This applies to the top method, choose which top k to focus on
#' @return Pairwise breakings, where the three columns are winner, loser and rank distance (latter used for Zemel)
#' @export
#' @examples
#' data(Data.Test)
#' Data.Test.pairs <- Breaking(Data.Test, "full")

def Breaking(Data):
    rows = Data.shape[0]
    cols = Data.shape[1]
    count = 0
    final = np.zeros((int(rows*cols*(cols - 1) / 2), 4), int)
    for i in range(0,len(Data)):
        current_list = Data[i]
        for j in range(0, len(current_list)):
            for k in range(j + 1, len(current_list)):
                final[count, 0] = current_list[j]
                final[count, 1] = current_list[k]
                final[count, 2] = k - j
                final[count, 3] = i + 1
                count = count + 1
    return final
#     m = Data.shape[1]

#   pair_full <- function(rankings) pair.top.k(rankings, length(rankings))

#   pair.top.k <- function(rankings, k) {
#     pair.top.helper <- function(first, rest) {
#       pc <- c();
#       z <- length(rest)
#       for(i in 1:z) pc <- rbind(pc, array(as.numeric(c(first, rest[i], i))))
#       pc
#     }
#     if(length(rankings) <= 1 | k <= 0) c()
#     else rbind(pair.top.helper(rankings[1], rankings[-1]), pair.top.k(rankings[-1], k - 1))
#   }

#   pair.adj <- function(rankings) {
#     if(length(rankings) <= 1) c()
#     else rbind(c(rankings[1], rankings[2]), pair.adj(rankings[-1]))
#   }

#   pair.top.partial <- function(rankings, m) {
#     # this is used in the case when we have missing ranks that we can
#     # fill in at the end of the ranking. We can assume here that all
#     # ranked items have higher preferences than non-ranked items
#     # (e.g. election data)

#     # the number of alternatives that are not missing
#     k <- length(rankings)

#     # these are the missing rankings
#     missing <- Filter(function(x) !(x %in% rankings), 1:m)

#     # if there is more than one item missing, scramble the rest and place them in the ranking
#     if(m - k > 1) missing <- scramble(missing)

#     # now just apply the top k breaking, with the missing elements
#     # inserted at the end
#     pair.top.k(c(rankings, missing), k)
#   }

#   break.into <- function(Data, breakfunction, ...) {
#     n <- nrow(Data)
#     # applying a Filter(identity..., ) to each row removes all of the missing data
#     # this is used in the case that only a partial ordering is provided
#     tmp <- do.call(rbind, lapply(1:n, function(row) cbind(breakfunction(Filter(identity, Data[row, ]), ...), row)))
#     colnames(tmp) <- c("V1", "V2", "distance", "agent")
#     tmp
#   }

#   if(method == "full") Data.pairs <- break.into(Data, pair.full)
#   if(method == "adjacent") Data.pairs <- break.into(Data, pair.adj)
#   if(method == "top") Data.pairs <- break.into(Data, pair.top.k, k = k)
#   if(method == "top.partial") Data.pairs <- break.into(Data, pair.top.partial, m = m)

#   Data.pairs
# }
