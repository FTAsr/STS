package com.shashi.dev;
import dkpro.similarity.algorithms.*;
import dkpro.similarity.algorithms.api.*;
import dkpro.similarity.algorithms.lexical.*;
import dkpro.similarity.algorithms.lexical.ngrams.WordNGramJaccardMeasure;
import dkpro.similarity.algorithms.lexical.string.GreedyStringTiling;

//import dkpro.similarity.algorithms.api.TextSimilarityMeasure;
//import dkpro.similarity.algorithms.lexical.ngrams.WordNGramJaccardMeasure;
import java.util.*;

public class Similarity {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// this similarity measure is defined in the dkpro.similarity.algorithms.lexical-asl package
		// you need to add that to your .pom to make that example work
		// there are some examples that should work out of the box in dkpro.similarity.example-gpl 
		TextSimilarityMeasure measure1 = new WordNGramJaccardMeasure(2);    // Use word trigrams
		TextSimilarityMeasure measure2 = new GreedyStringTiling(3);

		String[] tokens1 = "This is a short example text .".split(" ");   
		//String[] tokens2 = "A short example text could look like that .".split(" ");
		String[] tokens2 = "This is not a short example text .".split(" ");
		
		List<String> tok1 = new ArrayList<String>(Arrays.asList(tokens1));
		List<String> tok2 = new ArrayList<String>(Arrays.asList(tokens2));
		

		double score1 = 0, score2 = 0;
		try {
			//TextSimilarityMeasure(tokens1, tokens2);
			score1 = measure1.getSimilarity(tok1, tok2);
			score2 = measure2.getSimilarity(tok1, tok2);
		} catch (SimilarityException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		System.out.println("Similarity: " + score1);
		System.out.println("Similarity: " + score2);
	}

}
