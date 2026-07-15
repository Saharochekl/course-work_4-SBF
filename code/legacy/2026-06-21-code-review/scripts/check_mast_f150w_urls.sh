#!/usr/bin/env bash
set -u

printf "% -10s %-70s %-8s %s\n" "GALAXY" "FILE" "HTTP" "CONTENT_LENGTH"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o001_t001_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 1380' 'jw03055-o001_t001_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o002_t002_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 1399' 'jw03055-o002_t002_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o003_t003_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 1404' 'jw03055-o003_t003_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o006_t006_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 4472' 'jw03055-o006_t006_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o008_t008_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 4552' 'jw03055-o008_t008_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o014_t014_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 4636' 'jw03055-o014_t014_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o010_t010_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 4649' 'jw03055-o010_t010_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o011_t011_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 4697' 'jw03055-o011_t011_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o007_t007_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 4486' 'jw03055-o007_t007_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o004_t004_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 4374' 'jw03055-o004_t004_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o005_t005_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 4406' 'jw03055-o005_t005_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o009_t009_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 4621' 'jw03055-o009_t009_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o012_t012_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 1549' 'jw03055-o012_t012_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
status=$(curl -L -I --connect-timeout 10 --max-time 40 -s -o /tmp/mast_head.txt -w '%{http_code}' 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast%3AJWST%2Fproduct%2Fjw03055-o013_t013_nircam_clear-f150w_i2d.fits')
length=$(grep -i '^content-length:' /tmp/mast_head.txt | tail -1 | awk '{print $2}' | tr -d '\r')
printf '%-10s %-70s %-8s %s\n' 'NGC 3379' 'jw03055-o013_t013_nircam_clear-f150w_i2d.fits' "$status" "${length:-NA}"
