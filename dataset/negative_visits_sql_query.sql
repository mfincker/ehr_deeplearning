SELECT visit_id, patient_id, age_at_hosp_disch_in_days

FROM (  SELECT *
		FROM SHC_visit_de
		WHERE (hosp_disch_time > 0 
				AND contact_date > "2011-01-01")) AS negative_visits
ORDER BY rand()
LIMIT 641547;