import pageRankLinear
import pageRankPower
import CSVReader

def main():
    matrice_adjacencies = CSVReader.reader_csv("adjacences.csv")
    vector_personnalisation = CSVReader.reader_csv("VecteurPersonnalisation_Groupe30.csv")
    teleportation_factor = 0.9

    score_linear_method = pageRankLinear.pageRankLinear(matrice_adjacencies, teleportation_factor, vector_personnalisation)
    score_power_method = pageRankPower.pageRankPower(matrice_adjacencies, teleportation_factor, vector_personnalisation)

if __name__ == "__main__":
    main()
