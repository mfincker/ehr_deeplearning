Positive / Negative visits - database download
================
2018-02-22

DB and table connection
=======================

Connection to Stride7
---------------------

``` r
con <- 
  dbConnect(
  drv = RMySQL::MySQL(),
  dbname = "stride7",
  host = "shahlab-db1.stanford.edu",
  username = username,
  password = rstudioapi::askForPassword("Stride7 password"))
```

SHC\_visit\_de table
--------------------

``` r
SHC_visit_de <- tbl(con, "SHC_visit_de")
```

Visits
======

Visits with a discharge
-----------------------

``` r
visits_discharged <- 
  SHC_visit_de %>% 
  filter(hosp_disch_time > 0, 
         contact_date > "2011-01-01") 
```

ED visits
---------

``` r
visits_ed <- 
  SHC_visit_de %>% 
  filter(effective_dept_id == 2001002 || effective_dept_id == 999105,
         contact_date > "2011-01-01")
```

Positive visits
---------------

``` r
positive_visits <- 
  visits_discharged %>%
  inner_join(visits_ed, by = "patient_id", suffix = c("_discharge", "_ed")) %>%
  mutate(next_ed_visit = age_at_contact_in_days_ed - 
                         age_at_hosp_disch_in_days_discharge) %>%
  filter(next_ed_visit <= 3, next_ed_visit > 0) %>%
  select(patient_id, visit_id_discharge, age_at_hosp_disch_in_days_discharge) %>% 
  rename(visit_id = visit_id_discharge,
         age_at_hosp_disch_in_days = age_at_hosp_disch_in_days_discharge) %>% 
  distinct() %>% 
  mutate(label = 1)
```

Collect data

``` r
# positive_visits %>% 
#   collect() %>% 
#   write_tsv("positive_visits.tsv")
```

Negative visits
---------------

Instead of getting only negative visits, I downloaded a random bunch of discharged visits and removed the ones that are actually positive later (see [visits\_preprocessing.Rmd](./visits_preprocessing.Rmd)).

On dev-2, I ran:

``` bash
mysql -D stride7 < negative_visits_sql_query.sql > negative_visits_potential.txt
```

My SQL query was:

``` sql
SELECT visit_id, patient_id, age_at_hosp_disch_in_days

FROM (SELECT *
      FROM SHC_visit_de
      WHERE (hosp_disch_time > 0 AND contact_date > "2011-01-01")) 
      AS negative_visits
      
ORDER BY rand()
LIMIT 641547;
```
