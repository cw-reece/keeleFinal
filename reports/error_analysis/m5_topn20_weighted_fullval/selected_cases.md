# Selected Error Analysis Cases (weighted, topn=20, temp=2.0)

Generated file. Paste selected cases into the dissertation/poster.

## Improved cases

## Worsened cases

### QID 4398405 (img 439840) — worsened

**Q:** Where can i get a coffee table like this?

**GT answers:** ['furniture store', 'furniture store', 'furniture store', 'furniture store', 'store', 'store', 'walmart', 'walmart', 'goodwill', 'goodwill']

**Baseline:** walmart (soft=0.667, margin=0.000)

**Fused:** library (soft=0.000, margin=0.002, scale=0.0118)

**KG facts (preview):**
- get **RelatedTo** receive (score=3.2941818181818183)
- coffee_table **UsedFor** make_love (score=3.053)
- get **RelatedTo** acquire (score=2.773181818181818)
- get **RelatedTo** obtain (score=2.4881818181818183)
- coffee_table **IsA** table (score=2.25)
- coffee_table **Synonym** cocktail_table (score=2.2363636363636363)
- coffee_table **AtLocation** front_of_couch (score=2.2153846153846155)
- get **IsA** return (score=2.118181818181818)
- get **RelatedTo** getter (score=2.118181818181818)
- get **Synonym** acquire (score=2.118181818181818)

## High-confidence wrong baseline predictions

### QID 2085245 (img 208524) — same

**Q:** Does this type of train transport people or cargo?

**GT answers:** ['cargo', 'cargo', 'cargo', 'cargo', 'cargo', 'cargo', 'cargo', 'cargo', 'cargo', 'cargo']

**Baseline:** passenger (soft=0.000, margin=3.003)

**Fused:** passenger (soft=0.000, margin=3.006, scale=0.0118)

**KG facts (preview):**
- transport_people **CapableOf** bus (score=2.2363636363636363)
- transport_people **UsedFor** airplane (score=1.2363636363636363)
- transport_people **UsedFor** cars (score=1.2363636363636363)

### QID 1572615 (img 157261) — same

**Q:** What is this train hauling?

**GT answers:** ['cargo', 'cargo', 'cargo', 'cargo', 'cargo', 'cargo', 'coal', 'coal', 'people', 'people']

**Baseline:** passenger (soft=0.000, margin=2.864)

**Fused:** passenger (soft=0.000, margin=2.866, scale=0.0118)

**KG facts (preview):**
- train **CapableOf** arrive_late (score=4.61575)
- train **RelatedTo** express (score=3.317142857142857)
- train **AtLocation** human (score=2.157142857142857)
- train **IsA** boat_train (score=2.157142857142857)
- train **IsA** car_train (score=2.157142857142857)
- train **IsA** freight_train (score=2.157142857142857)
- train **IsA** hospital_train (score=2.157142857142857)
- train **IsA** mail_train (score=2.157142857142857)
- train **IsA** passenger_train (score=2.157142857142857)
- train **IsA** streamliner (score=2.157142857142857)

### QID 3849495 (img 384949) — same

**Q:** Which bridge is this?

**GT answers:** ['london bridge', 'london bridge', 'london bridge', 'london bridge', 'london', 'london', 'washington', 'washington', 'suspension', 'suspension']

**Baseline:** golden gate (soft=0.000, margin=2.783)

**Fused:** golden gate (soft=0.000, margin=2.784, scale=0.0118)

**KG facts (preview):**
- bridge **RelatedTo** water (score=5.902)
- bridge **RelatedTo** crossing (score=5.053999999999999)
- bridge **RelatedTo** over (score=5.0009999999999994)
- bridge **AtLocation** river (score=4.647)
- bridge **RelatedTo** river (score=4.43)
- bridge **AtLocation** trolls (score=4.175)
- bridge **UsedFor** crossing_river (score=4.157142857142857)
- bridge **RelatedTo** road (score=3.8569999999999998)
- bridge **AtLocation** homeless_person (score=3.621142857142857)
- bridge **UsedFor** cross_bay (score=3.621142857142857)

### QID 3795025 (img 379502) — same

**Q:** What type of cheese is on the pizza?

**GT answers:** ['ricotta', 'ricotta', 'ricotta', 'ricotta', 'feta', 'feta', 'feta', 'feta', 'motzarella', 'motzarella']

**Baseline:** mozzarella (soft=0.000, margin=2.618)

**Fused:** mozzarella (soft=0.000, margin=2.627, scale=0.0118)

**KG facts (preview):**
- cheese **IsA** food (score=6.45)
- cheese **CapableOf** age_well (score=5.410181818181818)
- pizza **IsA** disc_shaped_food_item (score=5.399692307692307)
- cheese **AtLocation** refrigerator (score=5.024)
- cheese **IsA** brie (score=4.597)
- cheese **AtLocation** pizza (score=4.266666666666667)
- pizza **AtLocation** cheese (score=4.266666666666667)
- pizza **AtLocation** oven (score=4.125)
- cheese **HasA** strong_odor (score=4.118181818181818)
- cheese **AtLocation** market (score=3.589)

