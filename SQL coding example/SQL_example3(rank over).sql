with tb1 as
(
select date as date1,
		away_team,
        away_score,
        rank() over(order by date) as date_rank 
from football
where away_team='Italy' and away_score<3
),
tb3 as
(
select date as date2,
        away_team,
        away_score,
        rank() over(order by date) as date_rank 
from football
where away_team='Italy' and away_score<3 
and date >(
		   select min(date) from football
		   where away_team='Italy' and away_score<3
		  )
)
select max(date2-date1) as max_time_window
from tb1 
inner join tb3 
on tb1.date_rank=tb3.date_rank;

