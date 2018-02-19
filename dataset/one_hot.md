# One Hot
Hershel Mehta  
February 11, 2018  



## R Markdown

# Setup Data

## Setup connection to stride7



## Setup diagnosis master table from stride7


```r
dx_master <- tbl(con, "dx_master")
```


```r
dx_master %>% head()
```

```
## # Source:   lazy query [?? x 8]
## # Database: mysql 10.1.30-MariaDB
## #   [hershelm@shahlab-db1.stanford.edu:/stride7]
##   patient_id  visit_id   contact_date_time AGE_AT_CONTACT_IN_DAYS    code
##        <dbl>     <dbl>               <chr>                  <dbl>   <chr>
## 1          1   9922763 2011-08-14 00:00:00                4335.00  139894
## 2          1 103602593 2015-10-16 08:29:00                5859.35 1359949
## 3          1 103602593 2015-10-16 08:29:00                5859.35 1298797
## 4          1 103602593 2015-10-16 08:29:00                5859.35 1298910
## 5          1  23782772 2014-12-12 00:00:00                5551.00  139894
## 6          1 101019436 2015-10-16 00:00:00                5859.35 1359949
## # ... with 3 more variables: code_source <chr>, table_source <chr>,
## #   dx_hospital <chr>
```

## 


```r
icd9_test <- 
  dx_master %>% 
  filter(
    #patient_ids from maevz,
    code_source == "ICD9CM",
    dx_hospital == "SHC"
  ) %>% 
  select(
    patient_id, age_in_days = AGE_AT_CONTACT_IN_DAYS, code
  ) %>% 
  mutate(
    n = 1
  ) %>% 
  head(100) %>% 
  collect()
```

```
## Warning in .local(conn, statement, ...): Decimal MySQL column 3 imported as
## numeric
```

## Sparse, One-Hot (each row is a single code)


```r
icd9_test %>% 
  distinct() %>% 
  mutate(
    id = rownames(.)
  ) %>% 
  spread(key = code, value = n, fill = 0)
```

```
## # A tibble: 99 x 73
##    patient_id age_in_days    id `233.0` `259.1` `274.0` `276.1` `276.8`
##  *      <dbl>       <dbl> <chr>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
##  1         12     8104.00    11       0       0       0       0       0
##  2         12     8104.00    17       0       0       0       0       0
##  3         12     8104.00    18       0       0       0       0       0
##  4         12     8104.00    20       0       0       0       0       0
##  5         12     8104.00     6       0       0       0       0       0
##  6         12     8104.00     7       0       0       0       0       0
##  7         12     8104.77    12       0       0       0       0       0
##  8         12     8104.77    19       0       0       0       0       0
##  9         12     8104.77    21       0       0       0       0       0
## 10         12     8111.00     1       0       0       0       0       0
## # ... with 89 more rows, and 65 more variables: `278.00` <dbl>,
## #   `311` <dbl>, `323.9` <dbl>, `333.99` <dbl>, `367.1` <dbl>,
## #   `382.9` <dbl>, `388.01` <dbl>, `401.1` <dbl>, `401.9` <dbl>,
## #   `424.1` <dbl>, `441.00` <dbl>, `470` <dbl>, `472.0` <dbl>,
## #   `473.0` <dbl>, `473.9` <dbl>, `478.19` <dbl>, `518.81` <dbl>,
## #   `530.11` <dbl>, `564.00` <dbl>, `565.0` <dbl>, `568.81` <dbl>,
## #   `569.0` <dbl>, `574.00` <dbl>, `574.10` <dbl>, `574.20` <dbl>,
## #   `575.12` <dbl>, `575.8` <dbl>, `593.9` <dbl>, `611.79` <dbl>,
## #   `620.8` <dbl>, `622.10` <dbl>, `623.8` <dbl>, `625.8` <dbl>,
## #   `625.9` <dbl>, `626.4` <dbl>, `628.9` <dbl>, `632` <dbl>,
## #   `633.10` <dbl>, `633.90` <dbl>, `633.91` <dbl>, `719.46` <dbl>,
## #   `724.00` <dbl>, `724.02` <dbl>, `724.2` <dbl>, `729.5` <dbl>,
## #   `780.6` <dbl>, `785.2` <dbl>, `786.05` <dbl>, `786.2` <dbl>,
## #   `787.2` <dbl>, `789.00` <dbl>, `799.0` <dbl>, `813.42` <dbl>,
## #   `922.4` <dbl>, E885.2 <dbl>, E917.0 <dbl>, V06.5 <dbl>, V22.2 <dbl>,
## #   V67.00 <dbl>, V70.0 <dbl>, V72.31 <dbl>, V72.83 <dbl>, V72.84 <dbl>,
## #   V76.12 <dbl>, V84.01 <dbl>
```

## Sparse (each row is a single day)


```r
icd9_test %>% 
  distinct() %>% 
  spread(key = code, value = n, fill = 0)
```

```
## # A tibble: 60 x 72
##    patient_id age_in_days `233.0` `259.1` `274.0` `276.1` `276.8` `278.00`
##  *      <dbl>       <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>    <dbl>
##  1         12     8104.00       0       0       0       0       0        0
##  2         12     8104.77       0       0       0       0       0        0
##  3         12     8111.00       0       0       0       0       0        0
##  4         12     8116.00       0       0       0       0       0        0
##  5         12     8653.00       0       0       0       0       0        0
##  6         12     8655.00       0       0       0       0       0        0
##  7         12     8655.78       0       0       0       0       0        0
##  8         12     8660.00       0       0       0       0       0        0
##  9         12     8720.00       0       0       0       0       0        0
## 10         77    27583.00       0       0       0       0       0        0
## # ... with 50 more rows, and 64 more variables: `311` <dbl>,
## #   `323.9` <dbl>, `333.99` <dbl>, `367.1` <dbl>, `382.9` <dbl>,
## #   `388.01` <dbl>, `401.1` <dbl>, `401.9` <dbl>, `424.1` <dbl>,
## #   `441.00` <dbl>, `470` <dbl>, `472.0` <dbl>, `473.0` <dbl>,
## #   `473.9` <dbl>, `478.19` <dbl>, `518.81` <dbl>, `530.11` <dbl>,
## #   `564.00` <dbl>, `565.0` <dbl>, `568.81` <dbl>, `569.0` <dbl>,
## #   `574.00` <dbl>, `574.10` <dbl>, `574.20` <dbl>, `575.12` <dbl>,
## #   `575.8` <dbl>, `593.9` <dbl>, `611.79` <dbl>, `620.8` <dbl>,
## #   `622.10` <dbl>, `623.8` <dbl>, `625.8` <dbl>, `625.9` <dbl>,
## #   `626.4` <dbl>, `628.9` <dbl>, `632` <dbl>, `633.10` <dbl>,
## #   `633.90` <dbl>, `633.91` <dbl>, `719.46` <dbl>, `724.00` <dbl>,
## #   `724.02` <dbl>, `724.2` <dbl>, `729.5` <dbl>, `780.6` <dbl>,
## #   `785.2` <dbl>, `786.05` <dbl>, `786.2` <dbl>, `787.2` <dbl>,
## #   `789.00` <dbl>, `799.0` <dbl>, `813.42` <dbl>, `922.4` <dbl>,
## #   E885.2 <dbl>, E917.0 <dbl>, V06.5 <dbl>, V22.2 <dbl>, V67.00 <dbl>,
## #   V70.0 <dbl>, V72.31 <dbl>, V72.83 <dbl>, V72.84 <dbl>, V76.12 <dbl>,
## #   V84.01 <dbl>
```

## Original approach to one-hot encoding


```r
icd9_test %>% 
  select(-n) %>% 
  distinct() %>% 
  modelr::model_matrix( ~ . -1) %>% 
  arrange(patient_id, age_in_days)
```

```
## # A tibble: 99 x 72
##    patient_id age_in_days code233.0 code259.1 code274.0 code276.1
##         <dbl>       <dbl>     <dbl>     <dbl>     <dbl>     <dbl>
##  1         12     8104.00         0         0         0         0
##  2         12     8104.00         0         0         0         0
##  3         12     8104.00         0         0         0         0
##  4         12     8104.00         0         0         0         0
##  5         12     8104.00         0         0         0         0
##  6         12     8104.00         0         0         0         0
##  7         12     8104.77         0         0         0         0
##  8         12     8104.77         0         0         0         0
##  9         12     8104.77         0         0         0         0
## 10         12     8111.00         0         0         0         0
## # ... with 89 more rows, and 66 more variables: code276.8 <dbl>,
## #   code278.00 <dbl>, code311 <dbl>, code323.9 <dbl>, code333.99 <dbl>,
## #   code367.1 <dbl>, code382.9 <dbl>, code388.01 <dbl>, code401.1 <dbl>,
## #   code401.9 <dbl>, code424.1 <dbl>, code441.00 <dbl>, code470 <dbl>,
## #   code472.0 <dbl>, code473.0 <dbl>, code473.9 <dbl>, code478.19 <dbl>,
## #   code518.81 <dbl>, code530.11 <dbl>, code564.00 <dbl>, code565.0 <dbl>,
## #   code568.81 <dbl>, code569.0 <dbl>, code574.00 <dbl>, code574.10 <dbl>,
## #   code574.20 <dbl>, code575.12 <dbl>, code575.8 <dbl>, code593.9 <dbl>,
## #   code611.79 <dbl>, code620.8 <dbl>, code622.10 <dbl>, code623.8 <dbl>,
## #   code625.8 <dbl>, code625.9 <dbl>, code626.4 <dbl>, code628.9 <dbl>,
## #   code632 <dbl>, code633.10 <dbl>, code633.90 <dbl>, code633.91 <dbl>,
## #   code719.46 <dbl>, code724.00 <dbl>, code724.02 <dbl>, code724.2 <dbl>,
## #   code729.5 <dbl>, code780.6 <dbl>, code785.2 <dbl>, code786.05 <dbl>,
## #   code786.2 <dbl>, code787.2 <dbl>, code789.00 <dbl>, code799.0 <dbl>,
## #   code813.42 <dbl>, code922.4 <dbl>, codeE885.2 <dbl>, codeE917.0 <dbl>,
## #   codeV06.5 <dbl>, codeV22.2 <dbl>, codeV67.00 <dbl>, codeV70.0 <dbl>,
## #   codeV72.31 <dbl>, codeV72.83 <dbl>, codeV72.84 <dbl>,
## #   codeV76.12 <dbl>, codeV84.01 <dbl>
```



